"""Picking the best checkpoint: the score each check gives, which checkpoints to keep, and when to stop early.

Each validation check adds a line to the run's leaderboard.json. The "pick best by" setting names the metric that ranks
them. Step 0 (the model before training) is shown as the starting point but never picked: the finished model is always
a trained checkpoint. Validation picks; the test split is only scored once, at the end, with the picked checkpoint, so
the reported test number isn't flattered by the picking.

This is the usual recipe (Hugging Face's metric_for_best_model with load_best_model_at_end, Keras's and Lightning's
EarlyStopping and ModelCheckpoint): a monitored metric with a direction, the best checkpoint kept and loaded at the end,
patience counted in checks without a gain, and a least gain that counts (their min_delta / early_stopping_threshold),
which anvil calls minGain.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

# setting -> (metric in the validation results, direction, label); anvil.tasks.decision_metrics says what they are
PICK_BY = {
    "overall": ("overall", "max", "Overall score"),
    "auroc": ("auroc", "max", "Mean AUROC"),
    "catch_rate": ("catch_rate", "max", "Catch rate"),
    "macro_f1": ("macro_f1", "max", "Mean macro-F1"),
    "mae": ("mae", "min", "Mean absolute error"),
    "calibration": ("ece", "min", "Calibration error"),
    "accuracy": ("accuracy", "max", "Accuracy"),
    "val_loss": ("val_loss", "min", "Validation loss"),
}
# The question type a setting needs in the validation data; the others work for any decision data.
NEEDS = {"auroc": "noul", "catch_rate": "noul", "macro_f1": "choice", "mae": "score"}
TYPE_NAMES = {"noul": "yes/no", "choice": "choice", "score": "score"}


LABELS = {key: label for key, _, label in PICK_BY.values()}  # metric -> label


def problem(pick_by: str, types: set[str] | None) -> str | None:
    """Why a decision run can't be picked by this setting, or None if it can. types: the question types in its
    validation data (None when that isn't known)."""
    if pick_by not in PICK_BY:
        return f"Unknown \"pick best by\" setting: {pick_by}."
    need = NEEDS.get(pick_by)
    if need and types is not None and need not in types:
        return (f"{PICK_BY[pick_by][2]} needs {TYPE_NAMES[need]} questions, and the validation data has none. "
                "Pick best by something else.")
    return None


def metric_of(pick_by: str) -> tuple[str, str]:
    key, direction, _ = PICK_BY.get(pick_by, PICK_BY["val_loss"])
    return key, direction


def score(metrics: dict, pick_by: str) -> float | None:
    key, _ = metric_of(pick_by)
    value = metrics.get(key)
    if value is None and key != "val_loss":  # e.g. no yes/no question had both answers: fall back to the loss
        return None
    return value


def load(run_dir: Path) -> dict:
    path = run_dir / "leaderboard.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"entries": [], "best_step": None}


def save(run_dir: Path, board: dict) -> None:
    path = run_dir / "leaderboard.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(board, indent=1))
    os.replace(tmp, path)


@contextmanager
def updating(run_dir: Path):
    """Load the leaderboard, let the caller change it, and save it, all under a lock: the trainer and the judge can
    both add checks, and neither may overwrite the other's."""
    with (run_dir / "leaderboard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        board = load(run_dir)
        yield board
        save(run_dir, board)


def record(board: dict, step: int, metrics: dict, pick_by: str, min_gain: float = 0.0) -> dict:
    """Add a check. Returns the updated board, whose best_step is the best check so far (never step 0).

    A check becomes the best only by beating the best so far by more than min_gain (the "least gain that counts"
    setting, in the score's own units; 0 counts any rise): a smaller rise is within what the validation sample's
    noise can do, so it neither moves the pick nor starts the early-stopping count again. The entry keeps the score's
    standard error (se) when the check worked one out, so the run page can show how sure the score is."""
    key, direction = metric_of(pick_by)
    value = score(metrics, pick_by)
    if value is None:  # the chosen metric doesn't exist for this data: rank by validation loss instead
        key, direction, value = "val_loss", "min", metrics.get("val_loss")
    entry = {"step": step, "score": value, "metric": key, "direction": direction}
    se = (metrics.get("se") or {}).get(key)
    if se is not None:
        entry["se"] = se
    board["entries"] = [e for e in board["entries"] if e["step"] != step]
    board["entries"].append(entry)
    board["entries"].sort(key=lambda e: e["step"])
    if step == 0:
        return board
    top = best(board)
    improved = top is None or (value is not None and top["score"] is not None and (
        value > top["score"] + min_gain if direction == "max" else value < top["score"] - min_gain))
    if improved:
        board["best_step"] = step
    return board


def best(board: dict) -> dict | None:
    """The best check so far: {step, score, metric, direction}, or None before the first trained check."""
    return next((e for e in board["entries"] if e["step"] == board.get("best_step")), None)


def ranked(board: dict) -> list[dict]:
    """Trained checkpoints, best first."""
    entries = [e for e in board["entries"] if e["step"] > 0 and e["score"] is not None]
    if not entries:
        return []
    direction = entries[0]["direction"]
    return sorted(entries, key=lambda e: (-e["score"] if direction == "max" else e["score"], e["step"]))


def checks_without_gain(board: dict, after: int = 0) -> int:
    """Checks in a row without a new best, counting only the ones after step `after`. Worked out from the checks
    themselves, so it doesn't matter in what order they arrived (the judge can lag behind)."""
    if board.get("best_step") is None:
        return 0
    return sum(1 for e in board["entries"] if e["step"] > max(board["best_step"], after))


def should_stop(board: dict, patience: int, after: int = 0) -> bool:
    return patience > 0 and board.get("best_step") is not None and checks_without_gain(board, after) >= patience


def stop_reason(board: dict, patience: int, after: int = 0, min_gain: float = 0.0) -> str | None:
    """Why training should stop early, or None to carry on. Read from the live leaderboard, so a better check that
    arrives later cancels it. Only the checks after step `after` count: before it the model is still warming up."""
    if not should_stop(board, patience, after):
        return None
    metric = best(board)["metric"]
    by = f" by at least {min_gain:g}" if min_gain > 0 else ""
    return f"{LABELS.get(metric, metric)} didn't improve{by} for {patience} checks in a row"


def min_gain(cfg: dict) -> float:
    """The run's least gain that counts (0 for runs from before the setting: any rise counted)."""
    return float(cfg["checkpoints"].get("minGain") or 0)


def keep(board: dict, keep_best: int, steps: list[int]) -> set[int]:
    """Checkpoint steps to keep: the best keep_best, the newest (for Resume), and any not checked yet."""
    judged = {e["step"] for e in board["entries"]}
    kept = {e["step"] for e in ranked(board)[:max(1, keep_best)]}
    if steps:
        kept.add(max(steps))
    kept |= {s for s in steps if s not in judged}
    return kept
