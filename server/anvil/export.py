"""Exporting a version of a run's model as GGUF, the file format of llama.cpp (and LM Studio, Ollama and the like).

The server starts it, on the CPU (exporting needs no GPU), after writing exports/<version>/request.json:

    python -m anvil.export /data/runs/<id> <version>        version: "final" or a checkpoint, "step-0001500"

1. A Hugging Face folder for that version. The finished model is merged already (model/merged). A LoRA checkpoint holds
   only the adapter, so it's merged into the base model first, in a temporary folder; a full fine-tuning checkpoint is
   a whole model. A Clef model's head files go next to it, which is how llama.cpp's converter knows it's Clef.
2. llama.cpp's converter writes a full-quality BF16 GGUF from it (for Clef: a decision GGUF, head included).
3. llama-quantize writes each smaller type from that one. The BF16 file is kept only if it was asked for.
4. anvil.json says what the files are and how to prompt the model.

Progress is in exports/<version>/status.json, and the tools' output in exports/<version>/log.txt.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

LLAMA_CPP = Path(os.environ.get("LLAMA_CPP_DIR", "/opt/llama.cpp"))
# Types offered, with roughly how many bits each weight takes (for size estimates on the screen).
QUANTS = {"BF16": 16.0, "Q8_0": 8.5, "Q6_K": 6.56, "Q5_K_M": 5.69, "Q4_K_M": 4.85}
VERSION = re.compile(r"^(final|step-\d{7})$")
CLEF_FILES = ("joint_head_config.json", "joint_head.safetensors")


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


class Status:
    def __init__(self, folder: Path):
        self.path = folder / "status.json"
        self.status = json.loads(self.path.read_text()) if self.path.exists() else {}

    def update(self, **fields) -> None:
        self.status.update(fields, updated=time.time())
        write_json(self.path, self.status)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "model"


def run(command: list[str], log: Path) -> None:
    """A tool, with its output in the export's log. Its last lines become the error if it fails."""
    with log.open("ab") as out:
        out.write(f"\n$ {' '.join(command)}\n".encode())
        out.flush()
        code = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT).returncode
    if code != 0:
        tail = log.read_text(errors="replace").strip().splitlines()[-6:]
        raise RuntimeError(f"{Path(command[1] if command[0] == sys.executable else command[0]).name} failed: "
                           + " | ".join(line.strip() for line in tail))


def hf_folder(cfg: dict, run_dir: Path, version: str, work: Path) -> tuple[Path, bool]:
    """The version as a Hugging Face folder the converter can read, and whether it's a temporary one to remove."""
    clef = cfg["train_as"] == "decision" and cfg["method"] == "clef"
    if version == "final":
        folder = run_dir / "model" / "merged"
        if not (folder / "config.json").exists():
            raise ValueError("This run has no finished model to export. Export one of its checkpoints instead.")
        return folder, False
    checkpoint = run_dir / "checkpoints" / version
    if not (checkpoint / "state.json").exists():
        raise ValueError(f"Checkpoint {version} is gone (it may have been pruned).")
    head = checkpoint / "head"
    temporary = work / "hf"
    shutil.rmtree(temporary, ignore_errors=True)
    if cfg["training_type"] != "lora":  # a whole model already: link its files, and the head beside them
        if not clef:
            return checkpoint / "model", False
        temporary.mkdir(parents=True)
        for f in (checkpoint / "model").iterdir():
            (temporary / f.name).symlink_to(f)
    else:
        merge_lora(cfg, checkpoint / "adapter", temporary)
    if clef:
        for name in CLEF_FILES:
            shutil.copy2(head / name, temporary / name)
    return temporary, True


def merge_lora(cfg: dict, adapter: Path, out: Path) -> None:
    """The base model with a LoRA checkpoint merged into it, saved with its tokenizer, on the CPU."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = cfg["model"]
    token = os.environ.get("HF_TOKEN") or None
    remote = bool(m.get("custom_code"))
    weights = m["base_model"] if m["kind"] == "adapter" else m["id"]  # the checkpoint's adapter replaces a base adapter
    tokenizer_from = m["id"] if m["kind"] != "adapter" or (Path(m["id"]) / "tokenizer_config.json").exists() else weights
    tok = AutoTokenizer.from_pretrained(tokenizer_from, token=token, trust_remote_code=remote)
    model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.bfloat16, token=token, trust_remote_code=remote)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(out)
    tok.save_pretrained(out)


def about(cfg: dict, run_dir: Path, version: str, quants: list[str]) -> dict:
    """What the files are, which version of the run they come from, how it scored, and how to prompt it."""
    info = {"run": run_dir.name, "name": cfg["name"], "version": version, "base": cfg["model"]["id"],
            "train_as": cfg["train_as"], "method": cfg["method"], "quants": quants,
            "made_with": f"llama.cpp {os.environ.get('LLAMA_CPP_VERSION', '')} (convert_hf_to_gguf.py, "
                         "llama-quantize)".replace("  ", " ")}
    try:
        board = json.loads((run_dir / "leaderboard.json").read_text())
    except (OSError, ValueError):
        board = {"entries": []}
    model_about = run_dir / "model" / "merged" / "anvil.json"
    picked = json.loads(model_about.read_text()) if model_about.exists() else {}
    step = picked.get("picked_step") if version == "final" else int(version[5:])
    entry = next((e for e in board["entries"] if e["step"] == step), None)
    info["step"] = step
    if entry:
        info["validation"] = {entry["metric"]: entry["score"]}
    if version == "final" and picked.get("final_test"):
        info["final_test"] = picked["final_test"]
    if cfg["train_as"] == "decision" and cfg["method"] == "clef":
        info["use"] = ("A decision GGUF (Clef): run it with llama-server, with batch sizes that fit your longest input "
                       "(a decision model reads it in one batch), e.g. `llama-server -m FILE -c 8192 -b 8192 -ub 8192`, "
                       "and POST a SystemOne request (model, state, questions) to /v1/systemone. Training trimmed each "
                       f"row to {cfg['hyper']['maxLength']} tokens; llama-server reads the whole state.")
    elif cfg["train_as"] == "decision":
        info["use"] = ("Prompt with the state, then '\\n\\nQuestion: <instructions>', the options as 'A. ...' lines, "
                       "then 'Answer:'; the next token's scores for ' A', ' B', ... are the answer.")
    return info


def export(run_dir: Path, version: str, status: Status) -> None:
    cfg = json.loads((run_dir / "config.json").read_text())
    folder = run_dir / "exports" / version
    quants = json.loads((folder / "request.json").read_text())["quants"]
    log = folder / "log.txt"
    name = f"{slug(cfg['name'])}-{version}"

    status.update(state="running", stage="Getting the model ready" if version == "final" else
                  f"Merging {version} into the base model")
    source, temporary = hf_folder(cfg, run_dir, version, folder)
    try:
        full = folder / f"{name}-BF16.gguf"
        status.update(stage="Converting to GGUF (BF16)")
        run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(source), "--outtype", "bf16",
             "--outfile", str(full)], log)
        smaller = [q for q in quants if q != "BF16"]
        for i, quant in enumerate(smaller, 1):
            status.update(stage=f"Quantizing to {quant} ({i} of {len(smaller)})")
            run([str(LLAMA_CPP / "build" / "bin" / "llama-quantize"), str(full), str(folder / f"{name}-{quant}.gguf"),
                 quant], log)
        if "BF16" not in quants:
            full.unlink()
    finally:
        if temporary:
            shutil.rmtree(source, ignore_errors=True)
    write_json(folder / "anvil.json", about(cfg, run_dir, version, quants))
    order = list(QUANTS)  # biggest first, as on the screen
    found = sorted(folder.glob("*.gguf"), key=lambda f: next((i for i, q in enumerate(order) if f.stem.endswith(q)), 99))
    files = [{"name": f.name, "size": f.stat().st_size} for f in found]
    status.update(state="done", stage="Done", files=files, finished=time.time())


def main() -> None:
    run_dir, version = Path(sys.argv[1]), sys.argv[2]
    if not VERSION.match(version):
        raise SystemExit(f"not a version: {version}")
    status = Status(run_dir / "exports" / version)
    status.update(state="running", stage="Starting", pid=os.getpid(), error=None, started=time.time())
    try:
        export(run_dir, version, status)
    except BaseException as error:
        traceback.print_exc()
        message = str(error) if isinstance(error, (ValueError, RuntimeError)) else f"{type(error).__name__}: {error}"
        status.update(state="failed", stage="Failed", error=message)
        sys.exit(1)


if __name__ == "__main__":
    main()
