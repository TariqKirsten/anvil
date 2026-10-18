"""Runs: checking a run request, starting the trainer process, finding runs again, stopping and resuming them.

A run lives in ANVIL_DATA/runs/<id>/ (anvil.train describes the files). The server only reads those files and sends
signals. The trainer is its own process in its own session, so restarting the server doesn't stop a run.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import config, datasets, folders, gpus, models_check

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


def busy_gpus() -> dict[int, str]:
    busy = {}
    if runs_dir().exists():
        for path in runs_dir().iterdir():
            cfg = _read(path / "config.json")
            if cfg and status(path.name).get("state") in ACTIVE:
                busy[cfg["gpu"]] = path.name
    return busy


# ---------------------------------------------------------------- starting

def _choose_gpu(picked: list[int] | None) -> int:
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
    # One card for now (several at once is step 5). Prefer one that doesn't also drive a screen.
    candidates.sort(key=lambda g: (g["display"], g["index"]))
    return candidates[0]["index"]


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
    try:
        lr = float(h["learningRate"])
    except (TypeError, ValueError):
        raise ValueError("The learning rate isn't a number.") from None
    if not 0 < lr < 1:
        raise ValueError("Use a learning rate between 0 and 1.")
    for key, low in (("epochs", 1), ("batchSize", 1), ("gradAccum", 1), ("maxLength", 16)):
        if not isinstance(h.get(key), int) or h[key] < low:
            raise ValueError(f"{key} must be a whole number of at least {low}.")
    if int(h.get("maxSteps") or 0) < 0:
        raise ValueError("Max steps can't be negative.")


def create(form: dict, prepare: dict, token: str | None) -> str:
    train_as = form["trainAs"]
    method = form["decisionMethod"] if train_as == "decision" else form["lmFormat"]
    if train_as == "decision" and method == "clef":
        raise ValueError("Training the Clef head arrives in step 6. Pick Letters for now.")
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

    gpu = _choose_gpu(form.get("gpus"))
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
        "gpu": gpu,
        "hyper": form["hyper"],
        "checkpoints": form["checkpoints"],
        "created": time.time(),
    }
    _write(path / "config.json", cfg)
    _write(path / "status.json", {"state": "starting", "stage": "Starting", "step": 0, "created": cfg["created"]})
    _spawn(run_id, path, gpu, token, resume=False)
    return run_id


def _new_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "run"
    base = f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_id, n = base, 1
    while (runs_dir() / run_id).exists():
        run_id, n = f"{base}-{n}", n + 1
    return run_id


def _spawn(run_id: str, path: Path, gpu: int, token: str | None, resume: bool) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("ROCR_VISIBLE_DEVICES", "HF_TOKEN")}
    env.update(HIP_VISIBLE_DEVICES=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(SERVER_DIR),
               PYTHONUNBUFFERED="1")
    if token:
        env["HF_TOKEN"] = token
    log = (path / "log.txt").open("ab")
    log.write(f"\n===== {'resumed' if resume else 'started'} {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"on GPU {gpu} =====\n".encode())
    log.flush()
    _processes[run_id] = subprocess.Popen(
        [sys.executable, "-m", "anvil.train", str(path)] + (["--resume"] if resume else []),
        cwd=SERVER_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True)
    log.close()
    current = _read(path / "status.json", {}) or {}
    current.update(state="starting", stage="Starting", pid=_processes[run_id].pid, error=None, updated=time.time())
    _write(path / "status.json", current)


# ---------------------------------------------------------------- reading, stopping, resuming

def summary(run_id: str) -> dict:
    path = runs_dir() / run_id
    cfg = _read(path / "config.json", {}) or {}
    st = status(run_id)
    return {"id": run_id, "name": cfg.get("name", run_id), "train_as": cfg.get("train_as"), "method": cfg.get("method"),
            "model": (cfg.get("model") or {}).get("id"), "gpu": cfg.get("gpu"), "created": cfg.get("created"),
            "state": st.get("state"), "stage": st.get("stage"), "step": st.get("step", 0), "total": st.get("total"),
            "updated": st.get("updated"), "last_eval": st.get("last_eval"), "error": st.get("error")}


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
            "checkpoints": ckpts, "model": str(path / "model") if (path / "model").exists() else None}


def stop(run_id: str) -> None:
    path = run_dir(run_id)
    st = status(run_id)
    if st.get("state") not in ACTIVE:
        raise ValueError("This run isn't running.")
    os.kill(int(st["pid"]), signal.SIGTERM)  # the trainer saves a checkpoint after the current step, then exits
    st.update(stage="Stopping after this step", updated=time.time())
    _write(path / "status.json", st)


def resume(run_id: str, token: str | None) -> None:
    path = run_dir(run_id)
    st = status(run_id)
    if st.get("state") in ACTIVE:
        raise ValueError("This run is still running.")
    if st.get("state") == "done":
        raise ValueError("This run has finished.")
    cfg = _read(path / "config.json")
    if not Path(cfg["data"]["prepared"], "train.parquet").exists():
        raise ValueError("The prepared data for this run is gone. Prepare it again and start a new run.")
    busy = busy_gpus()
    if cfg["gpu"] in busy:
        raise ValueError(f"GPU {cfg['gpu']} is busy with {busy[cfg['gpu']]}. Resume when it's free.")
    _spawn(run_id, path, cfg["gpu"], token, resume=True)
