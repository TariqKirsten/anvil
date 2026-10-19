"""The training process. The server (anvil.runs) starts it as

    python -m anvil.train /data/runs/<id> [--resume]

with HIP_VISIBLE_DEVICES set to the chosen card, or, to train on several cards at once, once per card through
torchrun (see anvil.parallel). It reads the run's config.json, trains, and reports through files:
- status.json       what it's doing right now: state, stage, step of total, speed, ETA, memory, error
- metrics.jsonl     one line per logged step (training loss), per validation check, and the final test
- leaderboard.json  every check's score on the "pick best by" metric, and the best checkpoint so far (anvil.picking)
- log.txt           everything it prints (the server writes this file from the process output)
- checkpoints/      the trainable weights plus the optimizer's state: the best few, and the newest for Resume
- model/            the finished model, built from the best checkpoint: adapter/ and merged/ for LoRA, merged/ otherwise

Checks run here, pausing training briefly, unless the run has a judge (anvil.judge) on a spare GPU: then this process
only saves checkpoints, and watches the leaderboard the judge fills to know when to stop early.

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

from . import picking

if os.environ.get("ANVIL_SAFE_KERNELS") == "1":
    # The slow but plain paths, for models whose fast kernels misbehave. Must run before any model code is imported.
    sys.modules["fla"] = None
    sys.modules["causal_conv1d"] = None

VAL_ROWS = 2000  # validation checks use at most this many rows, so they take seconds, not minutes
TEST_ROWS = 5000  # the final test uses at most this many rows (about two minutes for a 2B decision model)
JUDGE_SILENT_S = 90  # a judge that hasn't reported for this long is taken to be gone
HEAD_LR = 1e-4  # Clef's head, unless the form says otherwise
HEAD_FIRST = 0.1  # the share of the steps a new Clef head trains alone, before the model joins in
MAX_SKIPS = 20  # batches skipped in a row (loss or gradients not numbers) before the run fails instead of going on
STOP = {"requested": False}


def _ask_to_stop(signum, frame):
    STOP["requested"] = True


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


class Reporter:
    def __init__(self, run_dir: Path, status_path: Path | None = None):
        self.dir = run_dir
        self.path = status_path or run_dir / "status.json"
        self.status = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.metrics = (run_dir / "metrics.jsonl").open("a")

    def update(self, **fields) -> None:
        self.status.update(fields, updated=time.time())
        write_json(self.path, self.status)

    def metric(self, entry: dict) -> None:
        self.metrics.write(json.dumps(entry) + "\n")
        self.metrics.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()
                       if not isinstance(v, dict)), flush=True)  # each question's scores are for the screen

    def fail(self, message: str) -> None:
        self.update(state="failed", stage="Failed", error=message, eta_s=None)


class Silent(Reporter):
    """The other cards of a multi-GPU run: process 0 reports for the run, so they say nothing, unless they fail."""

    def __init__(self, run_dir: Path, card: int | None):
        self.dir, self.path, self.card, self.status = run_dir, run_dir / "status.json", card, {}

    def update(self, **fields) -> None:
        pass

    def metric(self, entry: dict) -> None:
        pass

    def fail(self, message: str) -> None:
        try:
            status = json.loads(self.path.read_text())
        except (OSError, ValueError):
            status = {}
        where = f"GPU {self.card}" if self.card is not None else "Another card"
        status.update(state="failed", stage="Failed", error=f"{where}: {message}", eta_s=None, updated=time.time())
        write_json(self.path, status)


def _quiet_libraries() -> None:
    """The other cards would repeat every warning and progress bar process 0 already shows."""
    try:
        from transformers.utils import logging

        logging.set_verbosity_error()
        logging.disable_progress_bar()
    except ImportError:
        pass


def friendly(error: BaseException) -> str:
    import torch

    if isinstance(error, torch.OutOfMemoryError):
        return ("The GPU ran out of memory. Lower the batch size or the max length, or turn on gradient "
                "checkpointing, then resume or start again.")
    if isinstance(error, (ValueError, FileNotFoundError)):
        return str(error)
    return f"{error.__class__.__name__}: {error}"


# ---------------------------------------------------------------- the model and the data

def device_and_dtype(cfg: dict):
    import torch

    device = torch.device(os.environ.get("ANVIL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:  # with several cards, process N uses the Nth visible one
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    dtype = torch.bfloat16 if cfg["hyper"]["precision"] == "bf16" else torch.float16

    def autocast():
        return torch.autocast(device.type, dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()

    return device, dtype, autocast


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


def load_weights(model, task, checkpoint: Path, cfg: dict, device, dtype):
    """The model with a checkpoint's trainable weights: swapped in for LoRA, loaded afresh for full fine-tuning. The
    task's own weights (Clef's head) are swapped in too."""
    if (checkpoint / "head").exists():
        task.load(checkpoint / "head")
    if cfg["training_type"] == "lora":
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(checkpoint / "adapter" / "adapter_model.safetensors"),
                                                   device=str(device)))
        return model
    from transformers import AutoModelForCausalLM

    fresh = AutoModelForCausalLM.from_pretrained(
        str(checkpoint / "model"), dtype=dtype if device.type == "cuda" else None,
        trust_remote_code=bool(cfg["model"].get("custom_code"))).to(device)
    fresh.config.use_cache = False
    return fresh


def make_task(cfg: dict, tok, model=None, device=None):
    from . import tasks

    h = cfg["hyper"]
    if cfg["train_as"] != "decision":
        return tasks.LanguageModel(tok, cfg["data"]["format"], int(h["maxLength"]))
    if cfg["method"] == "clef":
        head, config = clef_head(cfg, model, device)
        task = tasks.DecisionClef(tok, int(h["maxLength"]), head, config)
    else:
        task = tasks.DecisionLetters(tok, int(h["maxLength"]), int(h["seed"]))
    task.false_alarm_rate = false_alarm_rate(cfg)
    return task


def false_alarm_rate(cfg: dict) -> float:
    """The catch rate's allowance: the share of no cases that may be flagged (the form's percent; 1% before it had one)."""
    return float(cfg["checkpoints"].get("falseAlarmPercent", 1)) / 100


def clef_release(cfg: dict) -> Path | None:
    """The folder of the base model when it's a Clef release (it ships joint_head_config.json and
    joint_head.safetensors), so training continues from its head; None for an ordinary model."""
    m = cfg["model"]
    if m.get("kind") != "decision":
        return None
    if m["source"] == "folder":
        return Path(m["id"])
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(m["id"], allow_patterns=["joint_head*"], token=os.environ.get("HF_TOKEN") or None))


def fresh_head(cfg: dict) -> bool:
    return cfg["train_as"] == "decision" and cfg["method"] == "clef" and cfg["model"].get("kind") != "decision"


def clef_head(cfg: dict, model, device):
    """Clef's head: the base model's own if it's a Clef release, otherwise a new one sized for the model. A new one
    starts from the run's seed, so every card and the judge build the same one."""
    import torch

    from . import tasks
    from .vendor.clef.joint_schema_model import JointSchemaHead

    release = clef_release(cfg)
    if release is not None:
        config = json.loads((release / "joint_head_config.json").read_text())
        head = JointSchemaHead(**config)
        task = tasks.DecisionClef(None, 0, head.to(device), config)
        task.load(release)
        return task.head, config
    _, output_embeddings = tasks.decoder_and_head(model)
    config = tasks.clef_head_config(int(output_embeddings.shape[1]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(cfg["hyper"]["seed"]))
        head = JointSchemaHead(**config)
    return head.to(device=device, dtype=torch.float32), config


def read_split(cfg: dict, name: str, cap: int | None):
    import pandas as pd

    df = pd.read_parquet(Path(cfg["data"]["prepared"]) / f"{name}.parquet")
    if cap and len(df) > cap:  # the same rows every time: the trainer and the judge must score the same sample
        df = df.sample(cap, random_state=int(cfg["hyper"]["seed"])).reset_index(drop=True)
    return df


# ---------------------------------------------------------------- checkpoints

def checkpoints(run_dir: Path) -> list[Path]:
    folder = run_dir / "checkpoints"
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("step-*") if not p.name.endswith(".tmp") and (p / "state.json").exists())


def step_of(path: Path) -> int:
    return int(path.name.split("-")[1])


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step-{step:07d}"


def save_checkpoint(run_dir: Path, model, tok, opt, sched, state: dict, lora: bool, task=None) -> Path:
    import torch

    final = checkpoint_dir(run_dir, state["step"])
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if lora:
        model.save_pretrained(tmp / "adapter")
    else:
        model.save_pretrained(tmp / "model")
        tok.save_pretrained(tmp / "model")
    if task is not None and task.head is not None:
        task.save(tmp / "head")
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "rng": {"python": random.getstate(), "torch": torch.get_rng_state()}}, tmp / "optimizer.pt")
    write_json(tmp / "state.json", state)  # written last: a checkpoint without it is incomplete
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    return final


def prune(run_dir: Path, keep_steps: set[int]) -> None:
    for path in checkpoints(run_dir):
        if step_of(path) not in keep_steps:
            shutil.rmtree(path, ignore_errors=True)


def judge_alive(run_dir: Path) -> bool:
    try:
        status = json.loads((run_dir / "judge" / "status.json").read_text())
    except (OSError, ValueError):
        return False
    return status.get("state") not in ("done", "failed") and time.time() - status.get("updated", 0) < JUDGE_SILENT_S


# ---------------------------------------------------------------- training

def train(cfg: dict, run_dir: Path, report: Reporter, resume: bool, group) -> None:
    import numpy as np
    import torch

    from . import tasks

    h, ck = cfg["hyper"], cfg["checkpoints"]
    seed = int(h["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device, dtype, autocast = device_and_dtype(cfg)
    group.start(device)
    say = print if group.main else (lambda *args, **kwargs: None)  # process 0 speaks for the run
    lora = cfg["training_type"] == "lora"
    pick_by = ck.get("pickBy", "val_loss")
    keep_best, patience = int(ck.get("keepBest", 3)), int(ck.get("patience", 0))
    judged_elsewhere = cfg.get("judge_gpu") is not None

    report.update(stage="Reading the data", training_finished=False)
    train_df = read_split(cfg, "train", None)
    val_df = read_split(cfg, "validation", VAL_ROWS)

    found = checkpoints(run_dir)
    resume_from = group.share(str(found[-1]) if resume and found else None)  # every card resumes from the same one
    resume_from = Path(resume_from) if resume_from else None
    report.update(stage="Loading the model (downloading it first if needed)")
    tok, model = load_model(cfg, device, dtype, resume_from)

    report.update(stage="Preparing batches")
    task = make_task(cfg, tok, model, device)
    if resume_from is not None and (resume_from / "head").exists():
        task.load(resume_from / "head")
    items, parsed = task.items(train_df)
    val_items, val_parsed = task.items(val_df)
    lengths = task.lengths(items, parsed)
    world = group.world
    per_card, accum = int(h["batchSize"]), int(h["gradAccum"])
    per_batch = per_card * world  # one batch is split across the cards: each takes batch size rows of it
    batches_per_epoch = math.ceil(len(items) / per_batch)
    steps_per_epoch = max(1, math.ceil(batches_per_epoch / accum))
    total = int(h.get("maxSteps") or 0) or steps_per_epoch * int(h["epochs"])

    backbone = [p for p in model.parameters() if p.requires_grad]
    head = task.parameters()  # Clef's head; nothing for the other methods
    groups = [{"params": backbone, "lr": float(h["learningRate"])}]
    if head:
        groups.append({"params": head, "lr": float(h.get("headLearningRate") or HEAD_LR)})
    opt = torch.optim.AdamW(groups, weight_decay=float(h["weightDecay"]))
    # A new head is random at first, so the model would learn from noise: the head trains alone for a while.
    head_first = int(total * HEAD_FIRST) if fresh_head(cfg) else 0

    def warmup(start: int) -> int:
        """The warmup's steps, in a schedule over the steps from start to the end."""
        return int(max(1, total - start) * float(h["warmupPercent"]) / 100)

    def schedule(start: int):
        """Warmup, then the chosen schedule, over the steps from start to the end (nothing before start)."""
        span = max(1, total - start)
        warm = warmup(start)

        def factor(step: int) -> float:
            if step < start:
                return 0.0
            step -= start
            if step < warm:
                return (step + 1) / warm
            progress = min(1.0, (step - warm) / max(1, span - warm))
            if h["scheduler"] == "cosine":
                return 0.5 * (1 + math.cos(math.pi * progress))
            if h["scheduler"] == "linear":
                return 1.0 - progress
            return 1.0

        return factor

    sched = torch.optim.lr_scheduler.LambdaLR(opt, [schedule(head_first)] + ([schedule(0)] if head else []))
    # Until the model has warmed up (after a new head's steps alone, if any), a pause in its gains says nothing yet:
    # early stopping counts the checks after that, or after where Keep training picked up if that's later.
    warm_until = head_first + warmup(head_first)
    count_after = max(warm_until, int(ck.get("countFrom") or 0))
    step = epoch = batch_index = 0
    if resume_from is not None:
        state = json.loads((resume_from / "state.json").read_text())
        saved = torch.load(resume_from / "optimizer.pt", map_location=device, weights_only=False)
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        random.setstate(saved["rng"]["python"])
        torch.set_rng_state(saved["rng"]["torch"].cpu())  # map_location moved it to the GPU; it must live on the CPU
        step, epoch, batch_index = state["step"], state["epoch"], state["batch_index"]
        if state.get("world", 1) != world:
            raise ValueError(f"This run trained on {state.get('world', 1)} GPUs, so it must resume on as many.")
        report.metric({"type": "resume", "step": step})
    learner = group.wrap(model, task)

    n_trainable = sum(p.numel() for p in backbone + head)
    n_all = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head)
    cards = f"{world} cards" if world > 1 else str(device)
    say(f"{len(items):,} training items, {len(val_items):,} validation items; {total:,} steps "
        f"({steps_per_epoch:,} per epoch, {per_batch * accum:,} items per step); training {n_trainable / 1e6:.1f}M "
        f"of {n_all / 1e9:.2f}B parameters on {cards}; checks {'on the judge GPU' if judged_elsewhere else 'here'}",
        flush=True)
    def training() -> str:
        return "Training the head alone" if step < head_first else "Training"

    report.update(state="running", stage=training(), step=step, total=total, epoch=epoch,
                  trainable_m=round(n_trainable / 1e6, 1), head_m=round(sum(p.numel() for p in head) / 1e6, 1),
                  head_first=head_first, warm_until=warm_until, items=len(items), world=world,
                  steps_per_epoch=steps_per_epoch, items_per_step=per_batch * accum)

    board = picking.load(run_dir)

    def check(at_step: int) -> None:
        """A validation check, shared across the cards, recorded on the leaderboard."""
        nonlocal board
        report.update(stage="Checking on validation")
        model.eval()
        out = task.outputs(model, val_items[group.rank::world], val_parsed, max(1, per_card * 2), autocast)
        model.train()
        metrics = task.metrics(task.combine(group.gather(out)), len(val_items))
        if not group.main:
            return
        report.metric({"type": "eval", "step": at_step, "by": "trainer", **metrics})
        with picking.updating(run_dir) as board:
            picking.record(board, at_step, metrics, pick_by)
        report.update(stage=training(), last_eval={"step": at_step, **metrics}, best_step=board.get("best_step"))

    def checkpoint() -> None:
        if not group.main:  # the copies are identical, so process 0 saves for every card
            return
        report.update(stage="Saving a checkpoint")
        save_checkpoint(run_dir, model, tok, opt, sched, {"step": step, "epoch": epoch, "batch_index": batch_index,
                                                          "world": world, "saved": time.time()}, lora, task)
        if not judged_elsewhere:  # with a judge, the judge keeps the best and deletes the rest
            prune(run_dir, picking.keep(board, keep_best, [step_of(p) for p in checkpoints(run_dir)]))
        report.update(stage=training(), checkpoints=[p.name for p in checkpoints(run_dir)])

    def early_stop_reason() -> str | None:
        return picking.stop_reason(picking.load(run_dir) if judged_elsewhere else board, patience, count_after)

    every = int(ck["every"])
    log_every = max(1, min(50, total // 200))
    if group.any(STOP["requested"]):  # Stop pressed while the model was loading: nothing trained, nothing to save
        report.update(state="stopped", stage="Stopped", step=step, stopped=time.time())
        group.end(wait=True)
        return
    if step == 0 and not judged_elsewhere and group.share(0 not in {e["step"] for e in board["entries"]}):
        check(0)  # the score before any training, as the starting point of the chart
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    # This step's loss, loss count and tokens on this card; the window's, all cards together, since the last log line.
    mine = [0.0, 0, 0]
    window = [0.0, 0, 0]
    skipped, micro, in_a_row = 0, 0, 0

    def skip(why: str) -> None:
        """Every card drops the step's batches so far, so none of them spoils the weights and they stay in step. A rare
        overflow costs a batch; batch after batch means the run can't go on, and it fails rather than spin forever."""
        nonlocal skipped, micro, in_a_row
        skipped, micro, in_a_row = skipped + 1, 0, in_a_row + 1
        opt.zero_grad(set_to_none=True)
        say(f"step {step + 1}: skipped batch {bi:,} of epoch {epoch}, {why}", flush=True)  # where to replay it from
        report.update(skipped=skipped)
        if in_a_row >= MAX_SKIPS:
            raise ValueError(f"Training can't go on after step {step:,}: {MAX_SKIPS} batches in a row had a loss or "
                             "gradients that weren't numbers (NaN or infinity). Resume continues from the newest "
                             "checkpoint.")

    # Speed and ETA come from recent steps only: the first steps include one-time kernel compiling, and validation
    # pauses aren't training speed.
    window_start, window_step, step_rate = time.time(), step, None
    finished, early = step >= total, None
    while not finished:
        batches = tasks.length_grouped_batches(lengths, per_batch, random.Random(seed + epoch))
        for bi in range(batch_index, len(batches)):
            if len(batches[bi]) < world:  # the data's last few rows: too few to give every card one
                continue
            last = micro + 1 >= accum or bi + 1 >= len(batches)  # the step's last batch: update the weights after it
            batch = task.collate([items[i] for i in batches[bi][group.rank::world]], parsed, shuffle=True)
            with group.accumulating(learner, sync=last):
                with autocast():
                    loss = learner(batch)
                if not group.all(bool(torch.isfinite(loss))):  # a rare half-precision overflow on some card
                    skip("its loss wasn't a number")
                    continue
                (loss / accum).backward()
            micro += 1
            mine[0] += float(loss.detach())
            mine[1] += 1
            mine[2] += batch["tokens"]
            if not last:
                continue
            micro = 0
            # Each on its own, so a new head's big gradients don't shrink the model's.
            norms = [torch.nn.utils.clip_grad_norm_(params, 1.0) for params in (backbone, head) if params]
            if not group.all(all(torch.isfinite(n) for n in norms)):
                # The loss was fine but the gradients weren't. Clipping would spread a NaN to every gradient, and the
                # optimizer into every weight: from then on every loss is NaN and every batch is skipped. Seen once
                # in 14,000 steps on two R9700s; the same steps replayed from the checkpoint before it trained cleanly,
                # so most likely a fault in the GPU stack rather than the data or the model's numbers.
                skip("its gradients weren't numbers (" + ", ".join(
                    f"{name} {float(n):.3g}" for name, n in zip(("model", "head"), norms)) + ")")
                continue
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            in_a_row = 0
            batch_index = bi + 1
            # The cards agree, every step, on whether to stop (Stop reaches process 0 only) and on the loss and tokens.
            judge_says = early_stop_reason() if group.main and judged_elsewhere else None
            agreed = group.sum([STOP["requested"], judge_says is not None, *mine])
            stop = agreed[0] > 0
            window = [w + a for w, a in zip(window, agreed[2:])]
            mine = [0.0, 0, 0]
            if group.main and (step % log_every == 0 or step == total):
                seconds = max(time.time() - window_start, 1e-9)
                rate = (step - window_step) / seconds
                step_rate = rate if step_rate is None else 0.7 * step_rate + 0.3 * rate
                entry = {"type": "train", "step": step, "epoch": epoch, "loss": window[0] / max(window[1], 1),
                         "lr": float(sched.get_last_lr()[0]), "tokens_per_s": window[2] / seconds}
                if head:  # Clef's head has its own learning rate (the model's is 0 while the head trains alone)
                    entry["head_lr"] = float(sched.get_last_lr()[1])
                if device.type == "cuda":
                    entry["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                report.metric(entry)
                report.update(stage=training(), step=step, epoch=epoch, loss=entry["loss"],
                              tokens_per_s=entry["tokens_per_s"],
                              peak_mem_gb=entry.get("peak_mem_gb"), skipped=skipped,
                              eta_s=(total - step) / step_rate if step_rate else None)
            if step % log_every == 0 or step == total:
                window = [0.0, 0, 0]
                window_start, window_step = time.time(), step
            at_checkpoint = step % every == 0 or step == total
            if at_checkpoint and judged_elsewhere and not group.share(judge_alive(run_dir) if group.main else None):
                # The judge crashed or hung: check here from now on, so checkpoints don't pile up unchecked and
                # early stopping still works. finish() checks the ones it missed.
                say("the judge stopped, so checks run here from now on", flush=True)
                judged_elsewhere, board = False, picking.load(run_dir)
            if at_checkpoint and not judged_elsewhere:
                check(step)
            if at_checkpoint or stop:
                checkpoint()
            if at_checkpoint:  # the check and the save aren't training time
                window[2] = 0
                window_start, window_step = time.time(), step
            if stop:
                report.update(state="stopped", stage="Stopped", step=step, eta_s=None, stopped=time.time())
                say(f"stopped at step {step}; resume continues from here", flush=True)
                group.end(wait=True)
                return
            if judge_says is not None or agreed[1] > 0:
                early = judge_says or "the judge's leaderboard says so"
            elif at_checkpoint and not judged_elsewhere:
                early = group.share(early_stop_reason() if group.main else None)
            if early or step >= total:
                finished = True
                break
        else:
            epoch += 1
            batch_index = 0
    if early:
        say(f"stopping early at step {step}: {early}", flush=True)
        report.metric({"type": "early_stop", "step": step, "reason": early})

    group.end(wait=True)  # every card stops at the same step; from here on process 0 works alone
    if group.main:
        finish(cfg, run_dir, report, model, tok, task, step, early, board, device, dtype, autocast)


def finish(cfg, run_dir: Path, report: Reporter, model, tok, task, step: int, early: str | None, board: dict,
           device, dtype, autocast) -> None:
    """Training is over: make sure every checkpoint has been checked, pick the best, score it once on the test split,
    and save it as the run's model."""
    ck, h = cfg["checkpoints"], cfg["hyper"]
    pick_by = ck.get("pickBy", "val_loss")
    lora = cfg["training_type"] == "lora"
    batch = max(1, int(h["batchSize"]) * 2)
    report.update(training_finished=True, eta_s=None)

    swapped = False
    if cfg.get("judge_gpu") is not None:
        report.update(stage="Waiting for the judge to check the last checkpoints")
        while True:
            board = picking.load(run_dir)
            judged = {e["step"] for e in board["entries"]}
            if all(step_of(p) in judged for p in checkpoints(run_dir)) or not judge_alive(run_dir):
                break
            time.sleep(2)
    judged = {e["step"] for e in board["entries"]}
    missing = [p for p in checkpoints(run_dir) if step_of(p) not in judged]
    if missing:  # the judge is gone: check what it didn't, here
        val_df = read_split(cfg, "validation", VAL_ROWS)
        val_items, val_parsed = task.items(val_df)
        for path in missing:
            report.update(stage=f"Checking {path.name} on validation")
            model = load_weights(model, task, path, cfg, device, dtype)
            swapped = True
            model.eval()
            metrics = task.evaluate(model, val_items, val_parsed, batch, autocast)
            report.metric({"type": "eval", "step": step_of(path), "by": "trainer", **metrics})
            with picking.updating(run_dir) as board:
                picking.record(board, step_of(path), metrics, pick_by)

    best = board.get("best_step") or step
    if (best != step or swapped) and checkpoint_dir(run_dir, best).exists():
        report.update(stage=f"Loading the best checkpoint (step {best})")
        model = load_weights(model, task, checkpoint_dir(run_dir, best), cfg, device, dtype)
    prune(run_dir, picking.keep(board, int(ck.get("keepBest", 3)), [step_of(p) for p in checkpoints(run_dir)]))

    report.update(stage="Final test on the test split")
    test_df = read_split(cfg, "test", TEST_ROWS)
    final_test = None
    if len(test_df):
        test_items, test_parsed = task.items(test_df)
        model.eval()
        final_test = {"step": best, **task.evaluate(model, test_items, test_parsed, batch, autocast)}
        report.metric({"type": "test", **final_test})

    report.update(state="saving", stage="Saving the model", final_test=final_test, best_step=best)
    out = run_dir / "model"
    shutil.rmtree(out, ignore_errors=True)
    about = {"train_as": cfg["train_as"], "method": cfg["method"], "format": cfg["data"]["format"],
             "base": cfg["model"]["id"], "run": run_dir.name, "picked_step": best, "picked_by": pick_by,
             "final_test": final_test}
    if cfg["train_as"] == "decision" and cfg["method"] == "clef":
        about["prompt"] = ("Clef: the state and every question in one prompt (encode_record in anvil/vendor/clef); "
                           "joint_head_config.json and joint_head.safetensors are the head, as in Cloudflare's releases")
    elif cfg["train_as"] == "decision":
        about["prompt"] = "state, then '\\n\\nQuestion: ...', options as 'A. ...' lines, then 'Answer:'; read the letters"
    if lora:
        model.save_pretrained(out / "adapter")
        task.save(out / "adapter")
        write_json(out / "adapter" / "anvil.json", about)
        model = model.merge_and_unload()
    model.save_pretrained(out / "merged")
    tok.save_pretrained(out / "merged")
    task.save(out / "merged")
    write_json(out / "merged" / "anvil.json", about)
    report.update(state="done", stage="Done", step=step, finished=time.time(), model=str(out), early_stop=early,
                  checkpoints=[p.name for p in checkpoints(run_dir)])
    print(f"saved the step-{best} checkpoint to {out}", flush=True)


def main() -> None:
    import faulthandler

    faulthandler.enable(all_threads=True)  # a crash in native code still prints where every thread was
    from . import parallel

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    group = parallel.Group()
    if group.main:
        report = Reporter(run_dir)
    else:
        report = Silent(run_dir, card=(cfg.get("gpus") or [None] * group.world)[group.rank])
        _quiet_libraries()
    signal.signal(signal.SIGTERM, _ask_to_stop)
    signal.signal(signal.SIGINT, _ask_to_stop)
    report.update(state="loading", stage="Starting", pid=os.getpid(), error=None, early_stop=None,
                  started=report.status.get("started") or time.time(), stopped=None)
    try:
        train(cfg, run_dir, report, args.resume, group)
    except BaseException as error:  # everything ends in a status the screen can show
        traceback.print_exc()
        report.fail(friendly(error))
        sys.exit(1)
    finally:
        group.end()


if __name__ == "__main__":
    main()
