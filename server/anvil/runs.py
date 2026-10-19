"""Runs: checking a run request, starting the trainer process, finding runs again, stopping and resuming them.

A run lives in ANVIL_DATA/runs/<id>/ (anvil.train describes the files). The server only reads those files and sends
signals. The trainer is its own process in its own session, so restarting the server doesn't stop a run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import config, datasets, export, folders, gpus, models_check, picking

SERVER_DIR = Path(__file__).resolve().parents[1]
ACTIVE = {"starting", "loading", "running", "saving"}
RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
DECISION_FORMATS = {"decision"}
LM_FORMATS = {"chat", "pairs", "text"}

_processes: dict[str, subprocess.Popen] = {}


def runs_dir() -> Path:
    return config.settings.data / "runs"


def run_dir(run_id: str) -> Path:
    if not RUN_ID.match(run_id):
        raise FileNotFoundError("No such run.")
    path = runs_dir() / run_id
    if not (path / "config.json").exists():
        raise FileNotFoundError("No such run.")
    return path


def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _alive(run_id: str, pid: int | None) -> bool:
    proc = _processes.get(run_id)
    if proc is not None:
        return proc.poll() is None  # poll() also reaps the finished process
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def status(run_id: str) -> dict:
    """The run's status, corrected when its process died without saying so (a crash, or the container restarting)."""
    path = runs_dir() / run_id / "status.json"
    current = _read(path, {}) or {}
    if current.get("state") in ACTIVE and not _alive(run_id, current.get("pid")):
        current.update(state="failed", stage="Failed", eta_s=None,
                       error="The training process ended without finishing. The log may say why; "
                             "Resume continues from the last checkpoint.")
        _write(path, current)
    return current


def training_gpus(cfg: dict) -> list[int]:
    """The cards a run trains on (runs from before multi-GPU training name just one)."""
    return list(cfg.get("gpus") or [cfg["gpu"]])


def busy_gpus() -> dict[int, str]:
    """Cards used by anvil runs: the training cards while they train, and the judge's card while the judge runs (after
    Stop it finishes the checkpoints it hasn't checked yet, so it can outlive the trainer for a while)."""
    busy = {}
    if runs_dir().exists():
        for path in runs_dir().iterdir():
            cfg = _read(path / "config.json")
            if not cfg:
                continue
            if status(path.name).get("state") in ACTIVE:
                busy.update({gpu: path.name for gpu in training_gpus(cfg)})
            if _judge_running(path.name):
                busy[judge_status(path.name)["gpu"]] = path.name
    return busy


# ---------------------------------------------------------------- starting

def _choose_gpus(picked: list[int] | None, spare_checks: bool = True) -> tuple[list[int], int | None]:
    """The cards to train on, and a spare card for the judge (None: the training cards do the checks themselves).

    Every picked card that's free trains. The first one is process 0, which also saves checkpoints and finishes the run,
    so it's a card that doesn't drive a screen if there is one. The judge gets a free card you didn't pick, unless
    spare_checks is off (to keep that card free for another run)."""
    found = gpus.list_gpus()["gpus"]
    if not found:
        raise ValueError("There's no GPU to train on.")
    busy = busy_gpus()
    if picked is None:
        picked = [g["index"] for g in found if not g["busy"]]
    candidates = [g for g in found if g["index"] in picked and g["index"] not in busy]
    if not candidates:
        taken = [f"GPU {i} is training {busy[i]}" for i in picked if i in busy]
        raise ValueError("None of the picked GPUs is free" + (f" ({'; '.join(taken)})." if taken else "."))
    candidates.sort(key=lambda g: (g["display"], g["index"]))
    train = [g["index"] for g in candidates]
    spare = [g["index"] for g in found if g["index"] not in train and g["index"] not in busy and not g["busy"]]
    return train, min(spare) if spare and spare_checks else None


def _model(form: dict, token: str | None) -> dict:
    model = form["model"]
    if model["source"] == "hub":
        if not model.get("repo"):
            raise ValueError("Pick a base model first.")
        report, ident = models_check.check_repo(model["repo"], token), model["repo"]
    else:
        if not model.get("folder"):
            raise ValueError("Pick a base model first.")
        path = folders.inside_roots(model["folder"])
        report, ident = models_check.check_folder(path), str(path)
    if report.verdict == "bad":
        raise ValueError(f"The base model can't be trained: {report.message}")
    return {"source": model["source"], "id": ident, "kind": report.kind, "base_model": report.base_model,
            "architecture": report.architecture, "params": report.params,
            "custom_code": any(i.label == "Custom code" for i in report.items)}


def _check_hyper(h: dict) -> None:
    for key, name in (("learningRate", "learning rate"), ("headLearningRate", "head learning rate")):
        if key == "headLearningRate" and not h.get(key):
            continue  # only Clef has a head; the trainer's default applies when it's left out
        try:
            lr = float(h[key])
        except (TypeError, ValueError):
            raise ValueError(f"The {name} isn't a number.") from None
        if not 0 < lr < 1:
            raise ValueError(f"Use a {name} between 0 and 1.")
    for key, low in (("epochs", 1), ("batchSize", 1), ("gradAccum", 1), ("maxLength", 16)):
        if not isinstance(h.get(key), int) or h[key] < low:
            raise ValueError(f"{key} must be a whole number of at least {low}.")
    if int(h.get("maxSteps") or 0) < 0:
        raise ValueError("Max steps can't be negative.")


def _check_picking(checkpoints: dict, train_as: str, report: dict) -> dict:
    """The checkpoint settings, once "pick best by" is known to work for the data (a language model is always picked
    by its validation loss) and the catch rate's false alarm allowance is a sensible percent."""
    checkpoints = dict(checkpoints)
    if train_as == "lm":
        checkpoints["pickBy"] = "val_loss"
        return checkpoints
    questions = report.get("questions")  # missing from data prepared before reports listed them: not known
    types = {q["type"] for q in questions.values()} if questions is not None else None
    reason = picking.problem(checkpoints.get("pickBy", "val_loss"), types)
    if reason:
        raise ValueError(reason)
    percent = checkpoints.get("falseAlarmPercent", 1)
    if isinstance(percent, bool) or not isinstance(percent, (int, float)) or not 0 <= percent <= 50:
        raise ValueError("The false alarms allowed for the catch rate must be 0 to 50%.")
    return checkpoints


def create(form: dict, prepare: dict, token: str | None) -> str:
    train_as = form["trainAs"]
    method = form["decisionMethod"] if train_as == "decision" else form["lmFormat"]
    if form["trainingType"] == "qlora":
        raise ValueError("QLoRA (4-bit) comes later. Pick LoRA or Full for now.")
    _check_hyper(form["hyper"])
    model = _model(form, token)

    job = datasets.start(prepare, token, run=False)
    if job is None or job.state != "done":
        raise ValueError("Prepare the data first (the button in the Data section).")
    fmt = job.report["format"]
    if train_as == "decision" and fmt not in DECISION_FORMATS:
        raise ValueError(f"A decision model needs decision data, but this is {job.report['format_label'].lower()} data.")
    if train_as == "lm" and fmt not in LM_FORMATS:
        raise ValueError("A language model needs chat, prompt and completion, or text data, not decision data.")
    if train_as == "lm":
        method = fmt  # for language models the data decides: chat, prompt and completion, or plain text
    checkpoints = _check_picking(form["checkpoints"], train_as, job.report)

    train_gpus, judge_gpu = _choose_gpus(form.get("gpus"), form.get("spareChecks", True) is not False)
    run_id = _new_id(form.get("name") or "run")
    path = runs_dir() / run_id
    path.mkdir(parents=True)
    cfg = {
        "id": run_id,
        "name": form.get("name") or run_id,
        "train_as": train_as,
        "method": method,
        "training_type": form["trainingType"],
        "model": model,
        "data": {"prepared": str(config.settings.prepared / job.fingerprint), "format": fmt,
                 "rows": job.report["rows"], "fingerprint": job.fingerprint},
        "gpus": train_gpus,
        "gpu": train_gpus[0],
        "judge_gpu": judge_gpu,
        "hyper": form["hyper"],
        "checkpoints": checkpoints,
        "created": time.time(),
    }
    _write(path / "config.json", cfg)
    _write(path / "status.json", {"state": "starting", "stage": "Starting", "step": 0, "created": cfg["created"]})
    _spawn(run_id, path, train_gpus, token, resume=False, judge_gpu=judge_gpu)
    return run_id


def _new_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "run"
    base = f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_id, n = base, 1
    while (runs_dir() / run_id).exists():
        run_id, n = f"{base}-{n}", n + 1
    return run_id


def command(module: str, path: Path, cards: int, args: list[str]) -> list[str]:
    """One process, or with several cards one per card through torchrun (which also stops them all if one fails)."""
    if cards == 1:
        return [sys.executable, "-m", module, str(path)] + args
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={cards}",
            "-m", module, str(path)] + args


def _start(module: str, key: str, path: Path, cards: list[int], token: str | None, log_path: Path,
           args: list[str]) -> int:
    visible = ",".join(str(c) for c in cards)  # process N uses the Nth of these
    env = {k: v for k, v in os.environ.items() if k not in ("ROCR_VISIBLE_DEVICES", "HF_TOKEN")}
    env.update(HIP_VISIBLE_DEVICES=visible, CUDA_VISIBLE_DEVICES=visible, PYTHONPATH=str(SERVER_DIR),
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "4"))
    if token:
        env["HF_TOKEN"] = token
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    where = f"GPU {cards[0]}" if len(cards) == 1 else f"GPUs {', '.join(map(str, cards))}"
    log.write(f"\n===== {module} {'resumed' if '--resume' in args else 'started'} "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')} on {where} =====\n".encode())
    log.flush()
    _processes[key] = subprocess.Popen(
        command(module, path, len(cards), args), cwd=SERVER_DIR, env=env, stdout=log,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log.close()
    return _processes[key].pid


def _spawn(run_id: str, path: Path, gpus: list[int], token: str | None, resume: bool,
           judge_gpu: int | None = None) -> None:
    pid = _start("anvil.train", run_id, path, gpus, token, path / "log.txt", ["--resume"] if resume else [])
    current = _read(path / "status.json", {}) or {}
    current.update(state="starting", stage="Starting", pid=pid, error=None, updated=time.time())
    _write(path / "status.json", current)
    if judge_gpu is not None:
        (path / "judge").mkdir(exist_ok=True)
        judge_pid = _start("anvil.judge", f"{run_id}:judge", path, [judge_gpu], token, path / "judge" / "log.txt", [])
        _write(path / "judge" / "status.json", {"state": "starting", "stage": "Starting", "gpu": judge_gpu,
                                                "pid": judge_pid, "updated": time.time()})


# ---------------------------------------------------------------- reading, stopping, resuming

def summary(run_id: str) -> dict:
    path = runs_dir() / run_id
    cfg = _read(path / "config.json", {}) or {}
    st = status(run_id)
    return {"id": run_id, "name": cfg.get("name", run_id), "train_as": cfg.get("train_as"), "method": cfg.get("method"),
            "model": (cfg.get("model") or {}).get("id"), "gpu": cfg.get("gpu"),
            "gpus": training_gpus(cfg) if cfg.get("gpu") is not None else [], "created": cfg.get("created"),
            "state": st.get("state"), "stage": st.get("stage"), "step": st.get("step", 0), "total": st.get("total"),
            "updated": st.get("updated"), "last_eval": st.get("last_eval"), "error": st.get("error"),
            "best": picking.best(picking.load(path)), "size": size(path)}


def size(path: Path) -> int:
    """Bytes a run takes on disk: its checkpoints, model, metrics and logs."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:  # removed meanwhile (the trainer pruning a checkpoint)
                pass
    return total


def list_runs() -> list[dict]:
    if not runs_dir().exists():
        return []
    found = [summary(p.name) for p in runs_dir().iterdir() if (p / "config.json").exists()]
    return sorted(found, key=lambda r: r.get("created") or 0, reverse=True)


def detail(run_id: str, log_lines: int = 200) -> dict:
    path = run_dir(run_id)
    metrics = []
    if (path / "metrics.jsonl").exists():
        for line in (path / "metrics.jsonl").read_text().splitlines():
            try:
                metrics.append(json.loads(line))
            except ValueError:
                continue
    log = ""
    if (path / "log.txt").exists():
        with (path / "log.txt").open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 200_000))
            log = "\n".join(f.read().decode("utf-8", "replace").splitlines()[-log_lines:])
    ckpts = sorted(p.name for p in (path / "checkpoints").glob("step-*") if (p / "state.json").exists()) \
        if (path / "checkpoints").exists() else []
    return {"config": _read(path / "config.json", {}), "status": status(run_id), "metrics": metrics, "log": log,
            "checkpoints": ckpts, "model": str(path / "model") if (path / "model").exists() else None,
            "leaderboard": picking.load(path), "judge": judge_status(run_id), "size": size(path),
            "versions": versions(run_id), "exports": exports(run_id), "export_params": export_params(path)}


def _judge_running(run_id: str) -> bool:
    judge = judge_status(run_id)
    return judge is not None and judge.get("state") not in ("done", "failed")


def judge_status(run_id: str) -> dict | None:
    path = runs_dir() / run_id / "judge" / "status.json"
    current = _read(path)
    if not current:
        return None
    if current.get("state") not in ("done", "failed") and not _alive(f"{run_id}:judge", current.get("pid")):
        if current.get("pid") or time.time() - current.get("updated", 0) > 120:
            current.update(state="failed", stage="Failed", error="The judge stopped without finishing. The trainer "
                                                                  "checks whatever it missed itself.")
            _write(path, current)
    return current


# ---------------------------------------------------------------- exporting as GGUF (anvil.export)

def versions(run_id: str) -> list[dict]:
    """What can be exported: the finished model, and every checkpoint still kept, with its validation score."""
    path = run_dir(run_id)
    scores = {e["step"]: e for e in picking.load(path)["entries"]}
    found = []
    if (path / "model" / "merged" / "config.json").exists():
        about = _read(path / "model" / "merged" / "anvil.json", {}) or {}
        entry = scores.get(about.get("picked_step"))
        found.append({"version": "final", "step": about.get("picked_step"), "score": entry and entry["score"],
                      "metric": entry and entry["metric"]})
    if (path / "checkpoints").exists():
        for checkpoint in sorted((path / "checkpoints").glob("step-*")):
            if (checkpoint / "state.json").exists() and export.VERSION.match(checkpoint.name):
                entry = scores.get(int(checkpoint.name[5:]))
                found.append({"version": checkpoint.name, "step": int(checkpoint.name[5:]),
                              "score": entry and entry["score"], "metric": entry and entry["metric"]})
    return found


def export_params(path: Path) -> int | None:
    """How many weights an export holds: the finished model's own (and Clef's head), for the screen's size estimates.
    The base repo's count can be larger (Qwen3.5 repos include a vision tower that the text model leaves out)."""
    merged = path / "model" / "merged"
    files = list(merged.glob("*.safetensors")) if merged.exists() else []
    try:
        return sum(models_check.safetensors_params(f) for f in files) or None
    except (OSError, ValueError):
        return None


def exports(run_id: str) -> list[dict]:
    """Every export of this run: how far it got, and the files it made."""
    path = runs_dir() / run_id / "exports"
    found = []
    if path.exists():
        for folder in sorted(path.iterdir()):
            current = _read(folder / "status.json")
            if not current:
                continue
            if current.get("state") in ("starting", "running") and not _alive(f"{run_id}:export", current.get("pid")):
                current.update(state="failed", stage="Failed", error="The export stopped without finishing. The log "
                                                                       "in its folder may say why; try again.")
                _write(folder / "status.json", current)
            found.append({"version": folder.name, **current})
    return found


def _exporting(run_id: str) -> bool:
    return any(e.get("state") in ("starting", "running") for e in exports(run_id))


def start_export(run_id: str, version: str, quants: list[str]) -> None:
    path = run_dir(run_id)
    if status(run_id).get("state") in ACTIVE:
        raise ValueError("This run is still going: export it once it has finished or been stopped.")
    if version not in {v["version"] for v in versions(run_id)}:
        raise ValueError("That version of the run doesn't exist (any more).")
    chosen = [q for q in export.QUANTS if q in set(quants)]
    if not chosen:
        raise ValueError("Pick at least one type to export.")
    if _exporting(run_id):
        raise ValueError("An export of this run is still going. Wait for it to finish.")
    folder = path / "exports" / version
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / "request.json", {"quants": chosen, "requested": time.time()})
    env = {k: v for k, v in os.environ.items() if k not in ("ROCR_VISIBLE_DEVICES",)}
    env.update(HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(SERVER_DIR), PYTHONUNBUFFERED="1")
    log = (folder / "log.txt").open("ab")
    log.write(f"\n===== anvil.export {version} {', '.join(chosen)} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n".encode())
    log.flush()
    _processes[f"{run_id}:export"] = subprocess.Popen(
        [sys.executable, "-m", "anvil.export", str(path), version], cwd=SERVER_DIR, env=env, stdout=log,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log.close()
    _write(folder / "status.json", {"state": "starting", "stage": "Starting", "quants": chosen,
                                    "pid": _processes[f"{run_id}:export"].pid, "updated": time.time()})


def export_file(run_id: str, version: str, name: str) -> Path:
    """A finished export's file, for downloading: only .gguf files and anvil.json from that export's own folder."""
    folder = run_dir(run_id) / "exports" / version
    if not export.VERSION.match(version) or "/" in name or name.startswith(".") or not (
            name.endswith(".gguf") or name == "anvil.json"):
        raise FileNotFoundError("No such file.")
    file = folder / name
    if not file.is_file():
        raise FileNotFoundError("No such file.")
    return file


def delete(run_id: str) -> int:
    """Delete a run with everything it saved: checkpoints, the finished model, metrics and logs. Returns the bytes freed.
    A run that's still training, or whose judge is still checking, has to finish or be stopped first."""
    path = run_dir(run_id)
    if status(run_id).get("state") in ACTIVE:
        raise ValueError("This run is still going. Stop it first, then delete it.")
    if _judge_running(run_id):
        raise ValueError("This run's judge is still checking its last checkpoints. Try again when it's done.")
    if _exporting(run_id):
        raise ValueError("This run is being exported. Try again when the export is done.")
    freed = size(path)
    shutil.rmtree(path)
    _processes.pop(run_id, None)
    _processes.pop(f"{run_id}:judge", None)
    _processes.pop(f"{run_id}:export", None)
    return freed


def stop(run_id: str) -> None:
    path = run_dir(run_id)
    st = status(run_id)
    if st.get("state") not in ACTIVE:
        raise ValueError("This run isn't running.")
    os.kill(int(st["pid"]), signal.SIGTERM)  # the trainer saves a checkpoint after the current step, then exits
    st.update(stage="Stopping after this step", updated=time.time())
    _write(path / "status.json", st)


def early_stopped(st: dict) -> bool:
    """Finished because early stopping said so, before its last step: Resume can carry on training it."""
    return st.get("state") == "done" and bool(st.get("early_stop")) and st.get("step", 0) < (st.get("total") or 0)


def resume(run_id: str, token: str | None, patience: int | None = None) -> None:
    """Carry on from the newest checkpoint. patience, when given, replaces Stop after no gain for, counted from here:
    how many checks in a row without a gain to wait (0 trains to the last step). A run that stopped early needs it, or
    it would stop again at once."""
    path = run_dir(run_id)
    st = status(run_id)
    if st.get("state") in ACTIVE:
        raise ValueError("This run is still running.")
    again = early_stopped(st)
    if st.get("state") == "done" and not again:
        raise ValueError("This run has finished.")
    if again and patience is None:
        raise ValueError("Say how many more checks without a gain to wait, or 0 to train to the last step.")
    if patience is not None and not 0 <= patience <= 1000:
        raise ValueError("Use 0 (never stop early) or a whole number up to 1000.")
    if again and _exporting(run_id):
        raise ValueError("This run is being exported. Try again when the export is done.")
    cfg = _read(path / "config.json")
    if not Path(cfg["data"]["prepared"], "train.parquet").exists():
        raise ValueError("The prepared data for this run is gone. Prepare it again and start a new run.")
    busy = busy_gpus()
    cards = training_gpus(cfg)  # the same cards again: the batches are split by how many there are
    taken = [f"GPU {gpu} is busy with {busy[gpu]}" for gpu in cards if gpu in busy]
    if taken:
        raise ValueError(f"{'; '.join(taken)}. Resume when {'it is' if len(taken) == 1 else 'they are'} free.")
    if patience is not None:
        cfg["checkpoints"]["patience"] = patience
        _write(path / "config.json", cfg)
        with picking.updating(path) as board:  # the count starts again, so a later gain restarts it at patience too
            board["since_best"] = 0
    if again:  # no longer finished: the judge mustn't take the old finish as the end of training
        _write(path / "status.json", {**st, "early_stop": None, "training_finished": False, "finished": None,
                                      "final_test": None})
    judge_gpu = cfg.get("judge_gpu")
    if _judge_running(run_id):  # still checking from before the stop: it carries on with the resumed trainer
        _spawn(run_id, path, cards, token, resume=True)
        return
    if judge_gpu is not None and judge_gpu in busy:  # its card is taken now: the training cards check instead
        judge_gpu = cfg["judge_gpu"] = None
        _write(path / "config.json", cfg)
    _spawn(run_id, path, cards, token, resume=True, judge_gpu=judge_gpu)
