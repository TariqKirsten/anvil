"""Uploading a version of a run to Hugging Face: its weights, its GGUF files, or both, to the account the token is for.

The server starts it, after writing uploads/<id>/request.json in the run's folder, with the account's token in HF_TOKEN:

    python -m anvil.upload /data/runs/<id> <upload id>

1. The weights, in Hugging Face format: the finished model (model/merged, or model/adapter for LoRA's adapter alone) or
   a checkpoint. A LoRA checkpoint is merged into the base model first (as anvil.export does), unless it's the adapter
   alone. A Clef model's head files go along, as Cloudflare's releases ship them.
2. GGUF files from the version's export (exports/<version>/), usually to a repo of their own, <name>-GGUF.
3. Each repo is created, private or public, or set to that if it exists. One commit per repo adds the files and a model
   card (README.md) saying what the model is, how it scored, how to use it and how it was trained. A card someone has
   edited on Hugging Face is left as it is: anvil only replaces a card it wrote itself.

Nothing on Hugging Face changes until a repo's commit lands, so an upload that fails or is cancelled halfway leaves
the repo as it was (apart from being created, or its visibility set). Cancel is the server ending this process outright
(anvil.runs.cancel_upload): a signal can't reach Python while a transfer runs in hf_xet's native code. Progress is in
uploads/<id>/status.json, the details in its log.txt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from . import export, picking

REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
UPLOAD_ID = re.compile(r"^\d{8}-\d{6}(-\d+)?$")
SKIP = {"anvil.json", "README.md"}  # anvil's own notes, with this machine's paths; the card replaces any README
CARD_MARK = re.compile(r"\n<!-- anvil ([0-9a-f]{16}) -->\n?$")
LETTERS = ("Prompt it with the state, then `\\n\\nQuestion: <instructions>`, the options as `A. ...` lines, then "
           "`Answer:`; the next token's scores for ` A`, ` B`, … are its answer.")
SCORES = [("overall", "Overall"), ("accuracy", "Accuracy"), ("auroc", "Mean AUROC (yes/no)"),
          ("catch_rate", "Catch rate (yes/no)"), ("false_alarms", "False alarms (yes/no)"),
          ("macro_f1", "Mean macro-F1 (choice)"), ("mae", "Mean absolute error (score)"),
          ("ece", "Calibration error"), ("val_loss", "Loss"), ("perplexity", "Perplexity")]


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


class Status:
    def __init__(self, folder: Path):
        self.path = folder / "status.json"
        self.status = read_json(self.path, {}) or {}
        self.lock = threading.Lock()  # progress arrives from the upload's own threads

    def update(self, **fields) -> None:
        with self.lock:
            self.status.update(fields, updated=time.time())
            write_json(self.path, self.status)


# ---------------------------------------------------------------- what goes up

def hub_id(value: str | None) -> str | None:
    """A Hugging Face repo id, or None for anything else (a folder on this machine has no place on a public card)."""
    return value if value and REPO_ID.match(value) and not value.startswith(".") else None


def weights_folder(cfg: dict, run_dir: Path, version: str, weights: str, work: Path) -> tuple[Path, list[Path], bool]:
    """The folder of the version's weights, the head files that go beside them (Clef's, for a checkpoint's adapter), and
    whether the folder is a temporary one to remove afterwards."""
    clef = cfg["train_as"] == "decision" and cfg["method"] == "clef"
    if weights == "adapter":
        if cfg["training_type"] != "lora":
            raise ValueError("Only a LoRA run has an adapter to upload on its own.")
        if version == "final":
            return run_dir / "model" / "adapter", [], False
        checkpoint = run_dir / "checkpoints" / version
        if not (checkpoint / "state.json").exists():
            raise ValueError(f"Checkpoint {version} is gone (it may have been pruned).")
        heads = [checkpoint / "head" / name for name in export.CLEF_FILES] if clef else []
        return checkpoint / "adapter", heads, False
    folder, temporary = export.hf_folder(cfg, run_dir, version, work)
    return folder, [], temporary


def folder_files(folder: Path, extra: list[Path]) -> list[tuple[str, Path]]:
    """The files to upload from a model folder, as (name in the repo, file here)."""
    if not folder.is_dir():
        raise ValueError("That version's weights aren't there any more.")
    found = [(p.name, p) for p in sorted(folder.iterdir()) if p.is_file() and p.name not in SKIP
             and not p.name.startswith(".")]
    return found + [(p.name, p) for p in extra if p.is_file()]


def gguf_files(run_dir: Path, version: str, names: list[str]) -> list[tuple[str, Path]]:
    folder = run_dir / "exports" / version
    found = []
    for name in names:
        if "/" in name or name.startswith(".") or not name.endswith(".gguf") or not (folder / name).is_file():
            raise ValueError(f"{name} isn't one of this version's GGUF files (any more). Export it again.")
        found.append((name, folder / name))
    return found


# ---------------------------------------------------------------- the model card

def _yaml_list(key: str, values: list[str]) -> str:
    return f"{key}:\n" + "".join(f"- {v}\n" for v in values)


def _size(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{round(n / 1e6)} MB" if n >= 1e6 else f"{max(1, round(n / 1e3))} kB"


def _num(value, digits=3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "–"


def scores_table(final_test: dict) -> str:
    rows = [(label, final_test[key]) for key, label in SCORES if isinstance(final_test.get(key), (int, float))]
    lines = ["| Score | Test set |", "|---|---|"] + [f"| {label} | {_num(v, 2 if label == 'Perplexity' else 3)} |"
                                                     for label, v in rows]
    return "\n".join(lines)


def questions_table(final_test: dict) -> str | None:
    questions = final_test.get("questions") or {}
    if not questions:
        return None
    main = {"noul": ("auroc", "AUROC"), "choice": ("macro_f1", "macro-F1"), "score": ("mae", "mean absolute error")}
    lines = ["| Question | Type | Score | Rows |", "|---|---|---|---|"]
    for name in sorted(questions):
        q, rows = questions[name], questions[name].get("rows")
        key, label = main.get(q.get("type"), ("accuracy", "accuracy"))
        kind = {"noul": "yes/no"}.get(q.get("type"), q.get("type", ""))
        lines.append(f"| `{name}` | {kind} | {label} {_num(q.get(key))} | {f'{rows:,}' if isinstance(rows, int) else '–'} |")
    return "\n".join(lines)


def card(cfg: dict, run_dir: Path, version: str, kind: str, repo: str, files: list[tuple[str, Path]],
         weights_repo: str | None = None) -> str:
    """The model card: what it is, how it scored, how to use it and how it was trained. kind: "merged", "adapter" or
    "gguf". weights_repo: for GGUF files, the repo the weights went to in the same upload, which they're quantized from."""
    m, h = cfg["model"], cfg["hyper"]
    decision, clef, lora = cfg["train_as"] == "decision", cfg["method"] == "clef", cfg["training_type"] == "lora"
    base = hub_id(m.get("base_model") if m.get("kind") == "adapter" else m.get("id")) or hub_id(m.get("base_model"))
    picked = read_json(run_dir / "model" / "merged" / "anvil.json", {}) or {}
    step = picked.get("picked_step") if version == "final" else int(version[5:])
    board = picking.load(run_dir)
    entry = next((e for e in board["entries"] if e["step"] == step), None)
    final_test = picked.get("final_test") if version == "final" else None

    meta = ""
    parent = weights_repo if kind == "gguf" and weights_repo else base
    if parent:  # GGUF files of weights uploaded beside them are their quantized form; on their own, a fine-tune's
        relation = "quantized" if parent == weights_repo else "adapter" if kind == "adapter" else "finetune"
        meta += f"base_model: {parent}\nbase_model_relation: {relation}\n"
    meta += f"library_name: {'gguf' if kind == 'gguf' else 'peft' if kind == 'adapter' else 'transformers'}\n"
    tags = ["anvil"] + (["lora"] if lora else []) + (["decision-model"] if decision else []) + (["clef"] if clef else [])
    tags += ["gguf", "llama.cpp"] if any(name.endswith(".gguf") for name, _ in files) else []
    meta += _yaml_list("tags", tags)
    if not decision:
        meta += "pipeline_tag: text-generation\n"

    what = (f"a decision model {'with a Clef head' if clef else 'that answers with the scores of letters'}" if decision
            else {"chat": "a chat model", "pairs": "a prompt-and-completion model"}.get(cfg["method"], "a language model"))
    origin = f"[{base}](https://huggingface.co/{base})" if base else "a base model"
    form = {"merged": "The weights, in Hugging Face format" + (" (LoRA merged into the base model)" if lora else ""),
            "adapter": "A LoRA adapter, to load on top of the base model",
            "gguf": "GGUF files for llama.cpp, LM Studio, Ollama and the like"}[kind]
    lines = [f"# {cfg['name']}", "",
             f"{what[0].upper()}{what[1:]}, fine-tuned from {origin} with anvil. {form}"
             + (f", quantized from [{weights_repo}](https://huggingface.co/{weights_repo})." if kind == "gguf" and weights_repo
                else ".")]
    if decision:
        lines += ["", "It reads a `state` and a set of `questions`, and answers each with probabilities: yes/no, one of "
                      "several choices, or a level on a score."]

    lines += ["", "## How it scored", ""]
    if step is None:
        lines += ["The finished model. It hasn't been scored yet."]
    elif final_test:
        rows = final_test.get("rows")
        lines += [f"The step {step:,} checkpoint, picked by its {picking.LABELS.get(entry['metric'], entry['metric']).lower() if entry else 'validation score'} "
                  f"on validation, scored once on {f'{rows:,} rows of ' if rows else ''}the test set, which played no part in "
                  "picking it:", "", scores_table(final_test)]
        per_question = questions_table(final_test)
        if per_question:
            lines += ["", "Each question on the test set:", "", per_question]
    elif entry and entry.get("score") is not None:
        lines += [f"The step {step:,} checkpoint. Its {picking.LABELS.get(entry['metric'], entry['metric']).lower()} on "
                  f"validation: {_num(entry['score'])}. It hasn't been scored on the test set (only the finished model is)."]
    else:
        lines += [f"The step {step:,} checkpoint. It hasn't been scored yet."]

    lines += ["", "## Use it", ""]
    ggufs = [(name, path) for name, path in files if name.endswith(".gguf")]
    if kind == "adapter":
        lines += ["```python", "from peft import PeftModel", "from transformers import AutoModelForCausalLM, AutoTokenizer",
                  "", f'base = AutoModelForCausalLM.from_pretrained("{base or "<the base model>"}")',
                  f'model = PeftModel.from_pretrained(base, "{repo}")',
                  f'tokenizer = AutoTokenizer.from_pretrained("{base or "<the base model>"}")', "```"]
        if clef:
            lines += ["", "The Clef head (`joint_head_config.json`, `joint_head.safetensors`) is beside the adapter: "
                          "Cloudflare's Clef code reads it."]
    elif kind == "merged" and clef:
        lines += ["Cloudflare's Clef code loads it with its head (`joint_head_config.json`, `joint_head.safetensors`) and "
                  "answers SystemOne requests:", "", "```python",
                  "from joint_schema_model import load_release_model, systemone", "",
                  f'model, processor = load_release_model("{repo}")',
                  "print(systemone(model, processor, request))  # request: model, state, questions", "```"]
    elif kind == "merged" and decision:
        lines += [LETTERS]
    elif kind == "merged":
        lines += ["```python", "from transformers import AutoModelForCausalLM, AutoTokenizer", "",
                  f'model = AutoModelForCausalLM.from_pretrained("{repo}")',
                  f'tokenizer = AutoTokenizer.from_pretrained("{repo}")', "```"]
        if cfg["method"] == "chat":
            lines += ["", "It learned the assistant's replies in the tokenizer's chat template: prompt it with "
                          "`tokenizer.apply_chat_template(messages, add_generation_prompt=True)`."]
    if ggufs:
        if kind != "gguf":
            lines += ["", "The GGUF files beside the weights run in llama.cpp, LM Studio, Ollama and the like:", ""]
        ggufs.sort(key=lambda found: -found[1].stat().st_size)  # biggest first, as on anvil's screen
        lines += ["| File | Size |", "|---|---|"] + [f"| `{name}` | {_size(path.stat().st_size)} |" for name, path in ggufs]
        quants = [name[:-5].rsplit("-", 1)[-1] for name, _ in ggufs]  # anvil names them <name>-<version>-<type>.gguf
        quant = next((q for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "BF16") if q in quants), None)
        serve = f"llama-server -hf {repo}{f':{quant}' if quant else ''}"
        if clef:
            lines += ["", "A decision GGUF with its Clef head: llama-server answers SystemOne requests (a `model` name, "
                          "the `state` and the `questions`) at `/v1/systemone`. A decision model reads its whole input in "
                          "one batch, so give it batch sizes that fit your longest input:", "",
                      "```bash", f"{serve} -c 8192 -b 8192 -ub 8192", "```"]
        elif decision:
            lines += ["", LETTERS]
        else:
            lines += ["", "```bash", serve, "```"]

    status = read_json(run_dir / "status.json", {}) or {}
    method = (f"LoRA (rank {h.get('loraRank')}, alpha {h.get('loraAlpha')})" if lora else "Full fine-tuning")
    rate = f"{h.get('learningRate')}, {h.get('scheduler')}, {h.get('warmupPercent')}% warmup"
    if clef and h.get("headLearningRate"):
        rate += f"; the head {h['headLearningRate']}"
    trained = None
    if step and status.get("items_per_step"):
        trained = f"{step:,} steps of {status['items_per_step']:,} rows"
        if version == "final" and status.get("step", step) > step:
            trained += f": the best checkpoint of {status['step']:,} steps"
    rows = (cfg.get("data") or {}).get("rows") or {}
    lines += ["", "## Training", "", "| | |", "|---|---|", f"| Method | {method} |",
              f"| Learning rate | {rate} |", f"| Max length | {int(h.get('maxLength', 0)):,} tokens |",
              f"| Precision | {h.get('precision')} |"]
    if trained:
        lines.append(f"| Trained | {trained} |")
    if isinstance(rows.get("train"), int):
        lines.append(f"| Data | {rows['train']:,} training rows, {rows.get('validation', 0):,} validation, "
                     f"{rows.get('test', 0):,} test |")
    body = f"---\n{meta}---\n\n" + "\n".join(lines) + "\n"
    return body + f"\n<!-- anvil {hashlib.sha256(body.encode()).hexdigest()[:16]} -->\n"


def ours(text: str) -> bool:
    """Whether a card is one anvil wrote and nobody has changed since: its mark is the hash of what's above it."""
    found = CARD_MARK.search(text)
    return bool(found) and hashlib.sha256(text[:found.start()].encode()).hexdigest()[:16] == found.group(1)


# ---------------------------------------------------------------- uploading

class Progress:
    """Stands in for huggingface_hub's console progress bars while a Xet upload runs: the bytes done so far go to the
    status instead. Anything else asked of it does nothing, so a change in the library costs the progress, not the
    upload."""

    report = None  # set to fn(done, total) around each commit

    def __init__(self, *args, **kwargs):
        pass

    def update_progress(self, total, items=None):
        try:
            if Progress.report:
                Progress.report(int(total.total_bytes_completed), int(total.total_bytes))
        except Exception:
            pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _watch_progress() -> None:
    try:
        from huggingface_hub.utils import _xet_progress_reporting

        _xet_progress_reporting.XetUploadProgressReporter = Progress
    except Exception:  # moved in a newer huggingface_hub: the stages still show
        pass


def describe(error: BaseException, repo: str) -> str:
    code = getattr(getattr(error, "response", None), "status_code", None)
    if code in (401, 403):
        return (f"Hugging Face refused to write to {repo}: the token can't write there. Save a write token in Settings "
                "(or one allowed to write to that organisation).")
    if code == 404:
        return f"There's no Hugging Face account or organisation {repo.split('/')[0]} the token can write to."
    if isinstance(error, (ValueError, RuntimeError)):
        return str(error)
    return f"{type(error).__name__}: {error}"


def upload(run_dir: Path, upload_id: str, status: Status, api) -> None:
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import RepositoryNotFoundError

    cfg = json.loads((run_dir / "config.json").read_text())
    folder = run_dir / "uploads" / upload_id
    want = json.loads((folder / "request.json").read_text())
    version, private = want["version"], bool(want["private"])
    visibility = "private" if private else "public"

    targets: dict[str, dict] = {}  # repo -> {"kind", "files"}
    temporary = None
    try:
        if want.get("weights"):
            status.update(stage="Getting the weights ready" if version == "final" or want["weights"] == "adapter"
                          else f"Merging {version} into the base model")
            source, heads, is_temporary = weights_folder(cfg, run_dir, version, want["weights"], folder)
            temporary = source if is_temporary else None
            targets[want["repo"]] = {"kind": want["weights"], "files": folder_files(source, heads)}
        if want.get("gguf"):
            files = gguf_files(run_dir, version, want["gguf"])
            if want["gguf_repo"] in targets:  # the same repo as the weights: the GGUF files go beside them
                targets[want["gguf_repo"]]["files"] += files
            else:
                targets[want["gguf_repo"]] = {"kind": "gguf", "files": files}
        total = sum(p.stat().st_size for t in targets.values() for _, p in t["files"])
        done_before = 0
        repos = [{"id": repo, "url": f"https://huggingface.co/{repo}", "private": private,
                  "files": [{"name": n, "size": p.stat().st_size} for n, p in t["files"]], "state": "waiting"}
                 for repo, t in targets.items()]
        status.update(state="running", bytes_total=total, bytes_done=0, repos=repos)
        last = [0.0]

        for i, (repo, target) in enumerate(targets.items()):
            entry = repos[i]
            entry["state"] = "running"
            status.update(stage=f"Getting {repo} ready", repos=repos)
            try:
                info = api.repo_info(repo)
            except RepositoryNotFoundError:
                info = None
            existed, was_private = info is not None, bool(info and info.private)
            if not existed:
                api.create_repo(repo, visibility=visibility, exist_ok=True)
            elif was_private != private:
                api.update_repo_settings(repo, visibility=visibility)
            entry.update(created=not existed, made=visibility if existed and was_private != private else None)

            weights_repo = next((r for r, t in targets.items() if t["kind"] != "gguf"), None) \
                if target["kind"] == "gguf" else None
            text = card(cfg, run_dir, version, target["kind"], repo, target["files"], weights_repo)
            keep_card = False
            if existed and any(getattr(s, "rfilename", "") == "README.md" for s in (getattr(info, "siblings", None) or [])):
                theirs = Path(api.hf_hub_download(repo, "README.md", cache_dir=str(folder / "cache")))
                keep_card = not ours(theirs.read_text(errors="replace"))
            entry["card"] = "kept" if keep_card else "written"
            operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                          for name, path in target["files"]]
            if not keep_card:
                operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=text.encode()))
            size = sum(p.stat().st_size for _, p in target["files"])

            def report(done: int, of: int, base=done_before) -> None:
                if time.time() - last[0] < 1 and done < of:
                    return
                last[0] = time.time()
                status.update(bytes_done=base + min(done, size), stage=f"Uploading to {repo}")

            Progress.report = report
            status.update(stage=f"Uploading to {repo}", repos=repos)
            what = "the weights" if target["kind"] != "gguf" else "GGUF files"
            commit = api.create_commit(
                repo, operations=operations, commit_message=f"{cfg['name']}: {what} from anvil ({version})",
                commit_description=f"Uploaded by anvil from run {cfg['name']}, version {version}.")
            Progress.report = None
            done_before += size
            entry.update(state="done", commit=getattr(commit, "commit_url", None))
            status.update(bytes_done=done_before, repos=repos)
    except BaseException:
        for entry in status.status.get("repos") or []:
            if entry.get("state") == "running":
                entry["state"] = "failed"
        raise
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(folder / "cache", ignore_errors=True)
    status.update(state="done", stage="Done", finished=time.time())


def main() -> None:
    from huggingface_hub import HfApi

    run_dir, upload_id = Path(sys.argv[1]), sys.argv[2]
    if not UPLOAD_ID.match(upload_id):
        raise SystemExit(f"not an upload: {upload_id}")
    status = Status(run_dir / "uploads" / upload_id)
    status.update(state="running", stage="Starting", pid=os.getpid(), error=None, started=time.time())
    _watch_progress()
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    want = read_json(run_dir / "uploads" / upload_id / "request.json", {}) or {}
    try:
        upload(run_dir, upload_id, status, api)
    except BaseException as error:
        traceback.print_exc()
        repo = next((r["id"] for r in status.status.get("repos") or [] if r.get("state") == "failed"),
                    want.get("repo") or want.get("gguf_repo") or "the repo")
        status.update(state="failed", stage="Failed", error=describe(error, repo))
        sys.exit(1)


if __name__ == "__main__":
    main()
