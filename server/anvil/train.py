"""The training process: one run, one GPU. The server (anvil.runs) starts it as

    python -m anvil.train /data/runs/<id> [--resume]

with HIP_VISIBLE_DEVICES set to the chosen card. It reads the run's config.json, trains, and reports through files:
- status.json      what it's doing right now: state, stage, step of total, speed, ETA, memory, error
- metrics.jsonl    one line per logged step (training loss) and per validation check
- log.txt          everything it prints (the server writes this file from the process output)
- checkpoints/     the trainable weights plus the optimizer's state, every N steps, to resume from
- model/           the finished model: adapter/ and merged/ for LoRA, merged/ for full fine-tuning

Stop sends SIGTERM: the run saves a checkpoint after the current step and exits, so stopping loses at most one step.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

if os.environ.get("ANVIL_SAFE_KERNELS") == "1":
    # The slow but plain paths, for models whose fast kernels misbehave. Must run before any model code is imported.
    sys.modules["fla"] = None
    sys.modules["causal_conv1d"] = None

VAL_ROWS = 2000  # validation checks use at most this many rows, so they take seconds, not minutes
KEEP_CHECKPOINTS = 2  # the newest ones; step 4 adds keeping the best
STOP = {"requested": False}


def _ask_to_stop(signum, frame):
    STOP["requested"] = True


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


class Reporter:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        path = run_dir / "status.json"
        self.status = json.loads(path.read_text()) if path.exists() else {}
        self.metrics = (run_dir / "metrics.jsonl").open("a")

    def update(self, **fields) -> None:
        self.status.update(fields, updated=time.time())
        write_json(self.dir / "status.json", self.status)

    def metric(self, entry: dict) -> None:
        self.metrics.write(json.dumps(entry) + "\n")
        self.metrics.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()), flush=True)


def friendly(error: BaseException) -> str:
    import torch

    if isinstance(error, torch.OutOfMemoryError):
        return ("The GPU ran out of memory. Lower the batch size or the max length, or turn on gradient "
                "checkpointing, then resume or start again.")
    if isinstance(error, (ValueError, FileNotFoundError)):
        return str(error)
    return f"{error.__class__.__name__}: {error}"


# ---------------------------------------------------------------- the model

def load_model(cfg: dict, device, dtype, checkpoint: Path | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m, h = cfg["model"], cfg["hyper"]
    token = os.environ.get("HF_TOKEN") or None
    remote = bool(m.get("custom_code"))
    lora = cfg["training_type"] == "lora"
    weights = m["base_model"] if m["kind"] == "adapter" else m["id"]
    if not lora and checkpoint is not None:
        weights = str(checkpoint / "model")  # full fine-tuning resumes from the saved weights themselves
    tokenizer_from = m["id"] if m["kind"] != "adapter" or (Path(m["id"]) / "tokenizer_config.json").exists() else weights
    tok = AutoTokenizer.from_pretrained(tokenizer_from, token=token, trust_remote_code=remote)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        weights, dtype=dtype if device.type == "cuda" else None, token=token, trust_remote_code=remote).to(device)
    model.config.use_cache = False
    if h["gradCheckpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if h["gradCheckpointing"]:
            model.enable_input_require_grads()
        adapter = checkpoint / "adapter" if checkpoint is not None else (m["id"] if m["kind"] == "adapter" else None)
        if adapter is not None:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=h["loraRank"], lora_alpha=h["loraAlpha"], lora_dropout=0.0, target_modules="all-linear",
                task_type="CAUSAL_LM"))
    else:
        model.requires_grad_(True)
    return tok, model


# ---------------------------------------------------------------- checkpoints

def checkpoints(run_dir: Path) -> list[Path]:
    folder = run_dir / "checkpoints"
    return sorted(p for p in folder.glob("step-*") if (p / "state.json").exists()) if folder.exists() else []


def save_checkpoint(run_dir: Path, model, tok, opt, sched, state: dict, lora: bool) -> Path:
    import torch

    final = run_dir / "checkpoints" / f"step-{state['step']:07d}"
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if lora:
        model.save_pretrained(tmp / "adapter")
    else:
        model.save_pretrained(tmp / "model")
        tok.save_pretrained(tmp / "model")
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "rng": {"python": random.getstate(), "torch": torch.get_rng_state()}}, tmp / "optimizer.pt")
    write_json(tmp / "state.json", state)  # written last: a checkpoint without it is incomplete
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    for old in checkpoints(run_dir)[:-KEEP_CHECKPOINTS]:
        shutil.rmtree(old, ignore_errors=True)
    return final


# ---------------------------------------------------------------- training

def train(cfg: dict, run_dir: Path, report: Reporter, resume: bool) -> None:
    import numpy as np
    import pandas as pd
    import torch

    from . import tasks

    h = cfg["hyper"]
    seed = int(h["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(os.environ.get("ANVIL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if h["precision"] == "bf16" else torch.float16
    amp = device.type == "cuda"

    def autocast():
        return torch.autocast(device.type, dtype=dtype) if amp else contextlib.nullcontext()

    report.update(stage="Reading the data")
    prepared = Path(cfg["data"]["prepared"])
    train_df = pd.read_parquet(prepared / "train.parquet")
    val_df = pd.read_parquet(prepared / "validation.parquet")
    if len(val_df) > VAL_ROWS:
        val_df = val_df.sample(VAL_ROWS, random_state=seed).reset_index(drop=True)

    resume_from = checkpoints(run_dir)[-1] if resume and checkpoints(run_dir) else None
    report.update(stage="Loading the model (downloading it first if needed)")
    tok, model = load_model(cfg, device, dtype, resume_from)
    lora = cfg["training_type"] == "lora"

    report.update(stage="Preparing batches")
    if cfg["train_as"] == "decision":
        task = tasks.DecisionLetters(tok, int(h["maxLength"]), seed)
    else:
        task = tasks.LanguageModel(tok, cfg["data"]["format"], int(h["maxLength"]))
    items, parsed = task.items(train_df)
    val_items, val_parsed = task.items(val_df)
    lengths = task.lengths(items, parsed)
    batch_size, accum = int(h["batchSize"]), int(h["gradAccum"])
    batches_per_epoch = math.ceil(len(items) / batch_size)
    steps_per_epoch = max(1, math.ceil(batches_per_epoch / accum))
    total = int(h.get("maxSteps") or 0) or steps_per_epoch * int(h["epochs"])

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(h["learningRate"]), weight_decay=float(h["weightDecay"]))
    warm = int(total * float(h["warmupPercent"]) / 100)

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total - warm))
        if h["scheduler"] == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        if h["scheduler"] == "linear":
            return 1.0 - progress
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    step = epoch = batch_index = 0
    if resume_from is not None:
        state = json.loads((resume_from / "state.json").read_text())
        saved = torch.load(resume_from / "optimizer.pt", map_location=device, weights_only=False)
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        random.setstate(saved["rng"]["python"])
        torch.set_rng_state(saved["rng"]["torch"].cpu())  # map_location moved it to the GPU; it must live on the CPU
        step, epoch, batch_index = state["step"], state["epoch"], state["batch_index"]
        report.metric({"type": "resume", "step": step})

    n_trainable = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{len(items):,} training items, {len(val_items):,} validation items; {total:,} steps "
          f"({steps_per_epoch:,} per epoch); training {n_trainable / 1e6:.1f}M of {n_all / 1e9:.2f}B parameters "
          f"on {device}", flush=True)
    report.update(state="running", stage="Training", step=step, total=total, epoch=epoch,
                  trainable_m=round(n_trainable / 1e6, 1), items=len(items))

    def validate() -> dict:
        report.update(stage="Checking on validation")
        model.eval()
        metrics = task.evaluate(model, val_items, val_parsed, max(1, batch_size * 2), autocast)
        model.train()
        report.metric({"type": "eval", "step": step, **metrics})
        report.update(stage="Training", last_eval={"step": step, **metrics})
        return metrics

    def checkpoint() -> None:
        report.update(stage="Saving a checkpoint")
        save_checkpoint(run_dir, model, tok, opt, sched,
                        {"step": step, "epoch": epoch, "batch_index": batch_index, "saved": time.time()}, lora)
        report.update(stage="Training", checkpoints=[p.name for p in checkpoints(run_dir)])

    every = int(cfg["checkpoints"]["every"])
    log_every = max(1, min(50, total // 200))
    if STOP["requested"]:  # Stop pressed while the model was loading: nothing trained, nothing to save
        report.update(state="stopped", stage="Stopped", step=step, stopped=time.time())
        return
    if step == 0:
        validate()  # the score before any training, as the starting point of the chart
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    losses, tokens, skipped, micro = [], 0, 0, 0
    # Speed and ETA come from recent steps only: the first steps include one-time kernel compiling, and validation
    # pauses aren't training speed.
    window_start, window_step, step_rate = time.time(), step, None
    finished = False
    while not finished:
        batches = tasks.length_grouped_batches(lengths, batch_size, random.Random(seed + epoch))
        for bi in range(batch_index, len(batches)):
            batch = task.collate([items[i] for i in batches[bi]], parsed, shuffle=True)
            with autocast():
                loss = task.loss(model, batch)
            if not torch.isfinite(loss):  # rare half-precision overflows: skip the batch, don't poison the weights
                skipped += 1
                opt.zero_grad(set_to_none=True)
                micro = 0
                continue
            (loss / accum).backward()
            micro += 1
            losses.append(float(loss.detach()))
            tokens += batch["tokens"]
            if micro < accum and bi + 1 < len(batches):
                continue
            micro = 0
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            batch_index = bi + 1
            if step % log_every == 0 or step == total:
                seconds = max(time.time() - window_start, 1e-9)
                rate = (step - window_step) / seconds
                step_rate = rate if step_rate is None else 0.7 * step_rate + 0.3 * rate
                entry = {"type": "train", "step": step, "epoch": epoch, "loss": float(np.mean(losses)),
                         "lr": float(sched.get_last_lr()[0]), "tokens_per_s": tokens / seconds}
                if device.type == "cuda":
                    entry["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                report.metric(entry)
                report.update(step=step, epoch=epoch, loss=entry["loss"], tokens_per_s=entry["tokens_per_s"],
                              peak_mem_gb=entry.get("peak_mem_gb"), skipped=skipped,
                              eta_s=(total - step) / step_rate if step_rate else None)
                losses, tokens = [], 0
                window_start, window_step = time.time(), step
            at_checkpoint = step % every == 0 or step == total
            if at_checkpoint:
                validate()
            if at_checkpoint or STOP["requested"]:
                checkpoint()
            if at_checkpoint:
                window_start, window_step, tokens = time.time(), step, 0
            if STOP["requested"]:
                report.update(state="stopped", stage="Stopped", step=step, eta_s=None, stopped=time.time())
                print(f"stopped at step {step}; resume continues from here", flush=True)
                return
            if step >= total:
                finished = True
                break
        else:
            epoch += 1
            batch_index = 0

    report.update(state="saving", stage="Saving the model", eta_s=None)
    out = run_dir / "model"
    shutil.rmtree(out, ignore_errors=True)
    about = {"train_as": cfg["train_as"], "method": cfg["method"], "format": cfg["data"]["format"],
             "base": cfg["model"]["id"], "run": run_dir.name}
    if cfg["train_as"] == "decision":
        about["prompt"] = "state, then '\\n\\nQuestion: ...', options as 'A. ...' lines, then 'Answer:'; read the letters"
    if lora:
        model.save_pretrained(out / "adapter")
        write_json(out / "adapter" / "anvil.json", about)
        model = model.merge_and_unload()
    model.save_pretrained(out / "merged")
    tok.save_pretrained(out / "merged")
    write_json(out / "merged" / "anvil.json", about)
    report.update(state="done", stage="Done", step=step, finished=time.time(), model=str(out))
    print(f"saved to {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    report = Reporter(run_dir)
    signal.signal(signal.SIGTERM, _ask_to_stop)
    signal.signal(signal.SIGINT, _ask_to_stop)
    report.update(state="loading", stage="Starting", pid=os.getpid(), error=None,
                  started=report.status.get("started") or time.time(), stopped=None)
    try:
        train(cfg, run_dir, report, args.resume)
    except BaseException as error:  # everything ends in a status the screen can show
        traceback.print_exc()
        report.update(state="failed", stage="Failed", error=friendly(error), eta_s=None)
        sys.exit(1)


if __name__ == "__main__":
    main()
