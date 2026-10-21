"""The judge: checks a run's checkpoints on a spare GPU while training carries on, undisturbed, on another.

The server starts it next to the trainer when a second card is free:

    python -m anvil.judge /data/runs/<id>      (with HIP_VISIBLE_DEVICES set to the spare card)

It builds the same model as the trainer once, scores it before training (step 0), then for every new checkpoint swaps
in that checkpoint's trainable weights and runs the same validation check the trainer would. Each score goes on the
leaderboard (anvil.picking), and the judge keeps the best checkpoints and deletes the rest. The trainer reads the same
leaderboard to decide when to stop early.

It reports in judge/status.json, with a heartbeat every 10 seconds so the trainer can tell it's still there, and it
exits once training has finished and every checkpoint has been checked.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from . import picking, train

STOP = {"requested": False}


def _ask_to_stop(signum, frame):
    STOP["requested"] = True


def run(cfg: dict, run_dir: Path, report: train.Reporter) -> None:
    ck, h = cfg["checkpoints"], cfg["hyper"]
    pick_by = ck.get("pickBy", "val_loss")
    keep_best = int(ck.get("keepBest", 3))
    device, dtype, autocast = train.device_and_dtype(cfg)

    report.update(state="loading", stage="Loading the model")
    tok, model = train.load_model(cfg, device, dtype, None)
    task = train.make_task(cfg, tok, model, device)
    val_items, val_parsed = task.items(train.read_split(cfg, "validation", train.VAL_ROWS))
    batch = max(1, int(h["batchSize"]) * 2)
    trained: dict = {}  # the trainer's trained-rows check (its trained_rows.json), read the first time a check needs it

    def trained_check(step: int) -> dict | None:
        if not trained:
            path = run_dir / "trained_rows.json"
            plan = json.loads(path.read_text()) if path.exists() else None
            if plan is None or step < plan["from"]:
                return None
            items, parsed = task.items(train.read_split(cfg, "train", None))
            trained.update(rows=[items[i] for i in plan["items"]], parsed=parsed, start=plan["from"])
        if step < trained["start"]:
            return None
        return train.trained_scores(task.evaluate(model, trained["rows"], trained["parsed"], batch, autocast))

    def score(step: int) -> None:
        report.update(state="checking", stage=f"Checking step {step}", step=step)
        model.eval()
        metrics = task.evaluate(model, val_items, val_parsed, batch, autocast)
        fit = trained_check(step)
        if fit:
            metrics["trained"] = fit
        metrics = train.with_samples(metrics, task.samples(model, val_items, val_parsed, autocast))
        report.metric({"type": "eval", "step": step, "by": "judge", **metrics})
        with picking.updating(run_dir) as board:
            picking.record(board, step, metrics, pick_by, picking.min_gain(cfg))
        train.prune(run_dir, picking.keep(board, keep_best, [train.step_of(p) for p in train.checkpoints(run_dir)]))

    judged = {e["step"] for e in picking.load(run_dir)["entries"]}
    if 0 not in judged:
        score(0)  # the model before training: LoRA starts as a no-op, so this is the base model
        judged.add(0)
    while not STOP["requested"]:
        pending = [p for p in train.checkpoints(run_dir) if train.step_of(p) not in judged]
        for path in pending:
            if not path.exists():  # deleted meanwhile (an older checkpoint the trainer no longer needs)
                continue
            model = train.load_weights(model, task, path, cfg, device, dtype)
            score(train.step_of(path))
            judged.add(train.step_of(path))
        trainer = json.loads((run_dir / "status.json").read_text())
        over = trainer.get("training_finished") or trainer.get("state") in ("done", "failed", "stopped")
        if over and all(train.step_of(p) in judged for p in train.checkpoints(run_dir)):
            break
        report.update(state="waiting", stage="Waiting for the next checkpoint", step=None)
        time.sleep(2)


def main() -> None:
    import faulthandler

    faulthandler.enable(all_threads=True)  # a crash in native code still prints where every thread was
    run_dir = Path(sys.argv[1])
    cfg = json.loads((run_dir / "config.json").read_text())
    (run_dir / "judge").mkdir(exist_ok=True)
    report = train.Reporter(run_dir, status_path=run_dir / "judge" / "status.json")
    signal.signal(signal.SIGTERM, _ask_to_stop)
    signal.signal(signal.SIGINT, _ask_to_stop)
    report.update(state="loading", stage="Starting", pid=os.getpid(), gpu=cfg.get("judge_gpu"), error=None)

    def heartbeat():
        while True:
            time.sleep(10)
            if report.status.get("state") in ("done", "failed"):
                return
            report.update()

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        run(cfg, run_dir, report)
        report.update(state="done", stage="Done", step=None)
    except BaseException as error:
        traceback.print_exc()
        report.update(state="failed", stage="Failed", error=train.friendly(error))
        sys.exit(1)


if __name__ == "__main__":
    main()
