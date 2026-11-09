"""Datasets: reading them, checking every row, removing copies, and cutting validation and test sets from train.

A dataset's format is recognised from its columns:
- decision: state, questions, gold (System One style, as JSON strings or objects)
- chat: messages, a list of {role, content}
- pairs: prompt, completion
- text: text

Every row is reduced to three parts, so one set of rules covers every format:
- input: what the model reads (the state, or the user's side of a chat). Copies and leaks are found on this.
- output: what is asked or written (the questions, or the assistant's side).
- label: the answer, for decision rows.

Two rows are copies when their inputs match and their outputs match, after lowercasing and joining up whitespace; when
the answers differ too, the copy is also a conflict. Near-copies are left to whatever made the data, which can judge
them on the text that varies (a pipeline's Dedupe step on the column it wrote, the guard builder's MinHash). Judged on
a whole input, text that every row shares outweighs the part that differs: "yup" and "i'm done" typed after the same
"Asked: ... | User typed:" counted as near-copies, and half a generated dataset was dropped.

Validation and test are cut from train whole groups at a time, so a group never sits on both sides: the rows with the
same input, and if you choose a column (the file's own, or a key of the meta a pipeline writes, such as the List value
a row was made from), the rows with the same value of it too. For decision data, each answer of each question also
gets its share of validation and test, the rarest answers first.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, folders

EXTENSIONS = (".parquet", ".jsonl", ".json", ".csv")
FORMATS = {
    "decision": ("state", "questions", "gold"),
    "chat": ("messages",),
    "pairs": ("prompt", "completion"),
    "text": ("text",),
}
LABELS = {"decision": "Decision", "chat": "Chat", "pairs": "Prompt and completion", "text": "Plain text"}
VERSION = 2  # bump when the preparation rules change, so data prepared under the old rules isn't reused
SPLITS = ("train", "validation", "test")
META = "meta"  # where a pipeline's Save puts every column of a row, as JSON

Progress = Callable[[str, float], None]


def _quiet(stage: str, fraction: float) -> None:
    pass


# ---------------------------------------------------------------- reading

def resolve_file(path: str) -> Path:
    resolved = folders.inside_roots(path, config.settings.all_dataset_roots)
    if not resolved.is_file():
        raise FileNotFoundError(f"{path} isn't a file.")
    if resolved.suffix.lower() not in EXTENSIONS:
        raise ValueError("anvil reads .parquet, .jsonl, .json and .csv files.")
    return resolved


def read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if suffix == ".json":
        try:
            return pd.read_json(path, dtype=False)
        except ValueError:
            return pd.read_json(path, lines=True, dtype=False)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    raise ValueError("anvil reads .parquet, .jsonl, .json and .csv files.")


def read_source(source: dict, token: str | None) -> pd.DataFrame:
    if source.get("kind") == "file":
        return read_file(resolve_file(source["path"]))
    if source.get("kind") == "hub":
        import datasets

        return datasets.load_dataset(source["repo"], source.get("config"), split=source["split"], token=token).to_pandas()
    raise ValueError("A dataset is a file on this machine or a Hugging Face dataset.")


def identity(source: dict, token: str | None) -> dict:
    """What makes a source's content what it is: changes to the file, or a new commit on the Hub, change this."""
    if source.get("kind") == "file":
        path = resolve_file(source["path"])
        stat = path.stat()
        return {"kind": "file", "path": str(path), "size": stat.st_size, "mtime": stat.st_mtime_ns}
    if source.get("kind") == "hub":
        from huggingface_hub import HfApi

        sha = HfApi().dataset_info(source["repo"], token=token).sha
        return {"kind": "hub", "repo": source["repo"], "config": source.get("config"), "split": source["split"],
                "sha": sha}
    raise ValueError("A dataset is a file on this machine or a Hugging Face dataset.")


def detect_format(columns) -> str | None:
    columns = set(columns)
    return next((name for name, needed in FORMATS.items() if set(needed) <= columns), None)


# ---------------------------------------------------------------- checking rows

def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _value(value):
    """A cell as plain Python: JSON strings parsed, numpy arrays as lists."""
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _flatten(value) -> str:
    """The text in a state, in order: {"message": "hi"} reads as "hi"."""
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return "" if value is None else str(value)


def _is_int(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def check_decision(state, questions, gold) -> tuple[str, str, str]:
    """(input, output, label) of a decision row. Raises ValueError with a plain reason when it breaks the format.

    The rules are the ones Unsloth Studio and System One apply, plus one of anvil's: an answer may be null, for a
    question the row asks without scoring it (a Clef prompt that asks every question while the row's label covers only
    some). At least one question needs an answer."""
    try:
        state, questions, gold = _value(state), _value(questions), _value(gold)
    except (ValueError, TypeError):
        raise ValueError("a column isn't valid JSON") from None
    if state in (None, "", {}, []):
        raise ValueError("the state is empty")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be an object holding at least one question")
    if not isinstance(gold, dict):
        raise ValueError("gold must be an object with an answer for each question")
    for name, question in questions.items():
        if not isinstance(question, dict) or question.get("type") not in ("noul", "choice", "score"):
            raise ValueError(f"question '{name}' needs the type noul, choice or score")
        extra = set(question) - {"type", "instructions", "criteria"}
        if extra:
            raise ValueError(f"question '{name}' may only have type, instructions and criteria, not {sorted(extra)}")
        if len(name) + len(_canonical(question)) > 20_000:
            raise ValueError(f"question '{name}' is longer than 20,000 characters")
        if name not in gold:
            raise ValueError(f"question '{name}' has no gold answer")
        answer = gold[name]
        if answer is None:  # asked, not scored
            continue
        if isinstance(answer, dict) and "label" in answer:  # a soft label: {"label": ..., "probabilities": {...}}
            answer = answer["label"]
        criteria = question.get("criteria")
        if question["type"] == "noul":
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
                raise ValueError(f"question '{name}' is a noul, so its criteria may only describe true and false")
            if not isinstance(answer, (bool, np.bool_)) and answer not in ("true", "false"):
                raise ValueError(f"question '{name}' is a noul, so its answer must be true or false")
        elif question["type"] == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError(f"question '{name}' is a choice, so its criteria must name 1 to 255 options")
            if not isinstance(answer, str) or answer not in criteria:
                raise ValueError(f"question '{name}': the answer {answer!r} isn't one of its options")
        else:
            if not isinstance(criteria, list) or not 1 <= len(criteria) <= 10:
                raise ValueError(f"question '{name}' is a score, so its criteria must list 1 to 10 levels")
            if not _is_int(answer) or not 0 <= answer < len(criteria):
                raise ValueError(f"question '{name}': the answer must be a level number from 0 to {len(criteria) - 1}")
    if all(gold[name] is None for name in questions):
        raise ValueError("every answer is null: at least one question needs an answer")
    return _flatten(state), _canonical(questions), _canonical(gold)


def check_chat(messages) -> tuple[str, str, str]:
    try:
        messages = _value(messages)
    except (ValueError, TypeError):
        raise ValueError("messages isn't valid JSON") from None
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a list with at least one message")
    reads, writes = [], []
    for message in messages:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise ValueError("every message needs a role and content")
        if message["role"] not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unknown role {message['role']!r}")
        content = message["content"]
        text = content if isinstance(content, str) else _canonical(content)
        (writes if message["role"] == "assistant" else reads).append(text)
    if not writes:
        raise ValueError("there's no assistant message to learn from")
    return " ".join(reads), "\n".join(writes), ""


def check_pairs(prompt, completion) -> tuple[str, str, str]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("the prompt is empty")
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("the completion is empty")
    return prompt, completion, ""


def check_text(text) -> tuple[str, str, str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the text is empty")
    return text, "", ""


CHECKS = {"decision": check_decision, "chat": check_chat, "pairs": check_pairs, "text": check_text}


@dataclass
class Parsed:
    inputs: list[str]
    outputs: list[str]
    labels: list[str]
    valid: np.ndarray
    examples: list[dict] = field(default_factory=list)

    @property
    def invalid(self) -> int:
        return int((~self.valid).sum())


def parse(df: pd.DataFrame, fmt: str, progress: Progress = _quiet, limit: int | None = None) -> Parsed:
    """Check every row (or the first `limit`) against the format."""
    n = len(df) if limit is None else min(limit, len(df))
    columns = [df[c].iloc[:n].tolist() for c in FORMATS[fmt]]
    check = CHECKS[fmt]
    inputs, outputs, labels = [""] * n, [""] * n, [""] * n
    valid = np.zeros(n, dtype=bool)
    examples = []
    for i, values in enumerate(zip(*columns)):
        try:
            inp, out, label = check(*values)
        except ValueError as error:
            if len(examples) < 5:
                examples.append({"row": i, "reason": str(error)})
            continue
        inputs[i], outputs[i], labels[i], valid[i] = normalize(inp), out, label, True
        if i % 50_000 == 0:
            progress("Checking rows", i / n)
    return Parsed(inputs, outputs, labels, valid, examples)


# ---------------------------------------------------------------- copies

def drop_copies(p: Parsed) -> tuple[np.ndarray, dict]:
    """Rows to keep: the first of every set of copies, the rows with the same input and the same output. A copy whose
    answer isn't the kept row's is also a conflict: the labels disagree."""
    keep = p.valid.copy()
    first: dict = {}
    counts = {"exact": 0, "conflicts": 0}
    for i in np.flatnonzero(p.valid):
        key = (p.inputs[i], p.outputs[i])
        if key in first:
            keep[i] = False
            counts["exact"] += 1
            counts["conflicts"] += first[key] != p.labels[i]
        else:
            first[key] = p.labels[i]
    return keep, counts


# ---------------------------------------------------------------- columns to keep rows together by

def _text(value) -> str | None:
    """A column's value as text to compare, or None when the row has none."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return _canonical(value.tolist() if isinstance(value, np.ndarray) else value)
    if value is None or pd.isna(value):
        return None
    return _canonical(value)


def extra_columns(frame: pd.DataFrame, fmt: str, only: str | None = None) -> dict[str, list]:
    """The columns beside the format's own, as text per row (None where a row has no value): the file's own columns,
    and the keys of the meta column a pipeline's Save writes (ask, style and so on) unless a column has that name.
    With only, just that one, if there is one."""
    taken = set(FORMATS[fmt])
    out = {str(c): [_text(v) for v in frame[c]] for c in frame.columns
           if c not in taken and c != META and only in (None, str(c))}
    if META in frame.columns and (only is None or (only not in out and only not in taken)):
        metas = []
        for value in frame[META]:
            try:
                value = _value(value)
            except (ValueError, TypeError):
                value = None
            metas.append(value if isinstance(value, dict) else {})
        for name in dict.fromkeys(str(k) for m in metas for k in m):
            if name not in out and name not in taken and only in (None, name):
                out[name] = [_text(m.get(name)) for m in metas]
    return out


def group_choices(frame: pd.DataFrame, fmt: str, labels: list[str]) -> list[dict]:
    """The columns these rows can be kept together by: the ones whose values repeat (one value in every row keeps
    everything together, a different one in each keeps nothing together). For decision data, a column that only says
    the answer again (the List of labels the rows were written for) is marked: keeping rows together by it would put
    every row with one answer on the same side of the split."""
    golds = {label for label in labels if label}
    choices = []
    for name, values in extra_columns(frame, fmt).items():
        present = [(v, label) for v, label in zip(values, labels) if v is not None]
        distinct = {v for v, _ in present}
        if len(distinct) < 2 or len(distinct) == len(present):
            continue
        answers: dict = {}
        for v, label in present:
            if label:
                answers.setdefault(v, set()).add(label)
        answer = bool(golds) and len(distinct) <= len(golds) and all(len(a) == 1 for a in answers.values())
        choices.append({"name": name, "values": len(distinct), "answer": answer})
    return choices


# ---------------------------------------------------------------- answers

def _answer_text(answer) -> str:
    if isinstance(answer, dict) and "label" in answer:  # a soft label
        answer = answer["label"]
    if isinstance(answer, (bool, np.bool_)):
        return "true" if answer else "false"
    return str(answer)


def answer_cells(label: str) -> list[tuple[str, str]]:
    """A decision row's (question, answer) pairs, from the gold check_decision wrote: what balancing shares out. A
    question the row asks without an answer (null) gives none."""
    return [(name, _answer_text(a)) for name, a in json.loads(label).items() if a is not None] if label else []


def answer_summary(splits: dict) -> dict:
    """{question: {answer: {split: rows}}} from {split: (each row's answer pairs, the split's rows)}. Score levels come
    in order, other answers most common first."""
    found: dict = {}
    for split, (cells, rows) in splits.items():
        for row in rows:
            for name, answer in cells[row]:
                found.setdefault(name, {}).setdefault(answer, dict.fromkeys(SPLITS, 0))[split] += 1

    def order(item):
        answer, counts = item
        return (0, int(answer), "") if answer.lstrip("-").isdigit() else (1, -sum(counts.values()), answer)

    return {name: dict(sorted(answers.items(), key=order)) for name, answers in sorted(found.items())}


def _said(answer: str) -> str:
    return {"true": "yes", "false": "no"}.get(answer, answer)


# ---------------------------------------------------------------- the split

def _half_up(x: float) -> int:
    return math.floor(x + 0.5)


def _together(*columns: list) -> list[int]:
    """Each row's group, when rows that share a value in any of these columns (None: no value) belong together."""
    parent = list(range(len(columns[0])))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for values in columns:
        first: dict = {}
        for i, value in enumerate(values):
            if value is not None:
                parent[find(i)] = find(first.setdefault(value, i))
    return [find(i) for i in range(len(parent))]


def split_groups(groups: list, rows: np.ndarray, val_fraction: float, test_fraction: float, seed: int,
                 answers: list | None = None) -> dict[str, np.ndarray]:
    """Cut validation and test from rows, whole groups at a time, in a shuffled but repeatable order.

    Each split wants its share of the rows, and each group goes to the split it brings closest to that, measured as a
    share of what the split wants: small groups fill validation and test up to their share and no further, and a big
    one goes where it fits. With answers (each row's (question, answer) pairs), every answer is shared out the same
    way, the rarest first, so an answer that few rows have still reaches validation and test: iterative stratification
    (Sechidis et al., 2011), with groups in place of rows. An answer whose share of a split rounds to no rows stays in
    train. A validation or test set that no group fits takes train's smallest."""
    fractions = {"validation": val_fraction, "test": test_fraction}

    def wanted(total: int) -> dict[str, int]:
        cut = {s: _half_up(total * f) for s, f in fractions.items()}
        return {"train": total - sum(cut.values()), **cut}

    def gain(need: dict, want: dict, split: str, n: int) -> float:
        """How much closer to what it wants a split gets with n more rows, as a share of what it wants."""
        return (abs(need[split]) - abs(need[split] - n)) / want[split] if want[split] > 0 else -math.inf

    members: dict = {}
    for row in rows:
        members.setdefault(groups[row], []).append(int(row))
    keys = list(members)
    keys = [keys[k] for k in np.random.default_rng(seed).permutation(len(keys))]
    want_rows = wanted(len(rows))
    need_rows = dict(want_rows)
    where: dict = {}  # the split each group went to

    def by_rows(split: str, n: int) -> tuple:  # ties go to the split that needs the most rows, then in SPLITS' order
        return gain(need_rows, want_rows, split, n), need_rows[split], -SPLITS.index(split)

    if answers is not None:
        cells: dict = {}
        total: dict = {}
        holding: dict = {}  # the groups with each answer, in the shuffled order
        for key in keys:
            counts: dict = {}
            for row in members[key]:
                for cell in answers[row]:
                    counts[cell] = counts.get(cell, 0) + 1
            cells[key] = counts
            for cell, n in counts.items():
                total[cell] = total.get(cell, 0) + n
                holding.setdefault(cell, []).append(key)
        want = {cell: wanted(n) for cell, n in total.items()}
        need = {cell: dict(w) for cell, w in want.items()}
        for cell in sorted(total, key=lambda c: (total[c], c)):
            for key in holding[cell]:
                if key in where:
                    continue
                size = len(members[key])
                split = max(SPLITS, key=lambda s: (gain(need[cell], want[cell], s, cells[key][cell]), *by_rows(s, size)))
                where[key] = split
                need_rows[split] -= size
                for c, n in cells[key].items():
                    need[c][split] -= n
    for key in keys:  # the groups with no answer to share out, or all of them without answers
        if key not in where:
            split = max(SPLITS, key=lambda s: by_rows(s, len(members[key])))
            where[key] = split
            need_rows[split] -= len(members[key])
    # A split that wants rows but got none, because every group is bigger than its share, takes train's smallest
    # group while train keeps another: far more rows than asked for beats a validation or test set with none.
    for split in ("validation", "test"):
        if want_rows[split] > 0 and split not in where.values():
            in_train = [key for key in keys if where[key] == "train"]
            if len(in_train) > 1:
                where[min(in_train, key=lambda k: len(members[k]))] = split
    out: dict[str, list] = {s: [] for s in SPLITS}
    for key in keys:
        out[where[key]] += members[key]
    return {name: np.array(sorted(r), dtype=np.int64) for name, r in out.items()}


# ---------------------------------------------------------------- preparing a dataset

INPUT_WORD = {"decision": "state", "chat": "prompt", "pairs": "prompt", "text": "text"}


def options(request: dict) -> dict:
    """The settings that change the result, with fixed types: 5 and 5.0 must give the same fingerprint."""
    keep = str(request.get("keep_together", "input"))
    if keep == "source":  # how the source column was asked for before any column could be
        keep = "column:source"
    return {"dedupe": bool(request.get("dedupe", True)),
            "val_percent": float(request.get("val_percent", 5)), "test_percent": float(request.get("test_percent", 5)),
            "keep_together": keep, "balance": bool(request.get("balance", True)),
            "seed": int(request.get("seed", 3407))}


def fingerprint(request: dict, identities: dict) -> str:
    blob = _canonical({"version": VERSION, "sources": identities, "options": options(request)})
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} or {items[-1]}"


def prepare(request: dict, token: str | None, out_dir: Path, progress: Progress = _quiet) -> dict:
    """Check, remove copies and split. Writes train/validation/test.parquet and report.json into out_dir."""
    started = time.time()
    opts = options(request)
    progress("Reading files", 0.0)
    frames = {name: read_source(request[name], token) for name in SPLITS if request.get(name)}
    fmt = detect_format(frames["train"].columns)
    if fmt is None:
        raise ValueError("Train's columns don't match a format anvil knows: decision (state, questions, gold), "
                         "chat (messages), prompt and completion, or text.")
    for name, frame in frames.items():
        if detect_format(frame.columns) != fmt:
            raise ValueError(f"The {name} data isn't in the same format as train ({LABELS[fmt]}).")

    parsed = {}
    for name, frame in frames.items():
        parsed[name] = parse(frame, fmt, lambda stage, f, n=name: progress(f"Checking {n} rows", f))

    notes: list[str] = []
    report: dict = {"format": fmt, "format_label": LABELS[fmt], "invalid": {}, "examples": {}, "removed": {},
                    "leaks": {}, "rows": {}, "notes": notes}
    keep = {}
    for name, p in parsed.items():
        report["invalid"][name] = p.invalid
        report["examples"][name] = p.examples
        if opts["dedupe"]:
            keep[name], report["removed"][name] = drop_copies(p)
        else:
            keep[name], report["removed"][name] = p.valid.copy(), {"exact": 0, "conflicts": 0}

    # Validation and test files you gave lose the rows whose input train has too, and test the ones validation has.
    train = parsed["train"]
    if opts["dedupe"]:
        seen = {t for t, ok in zip(train.inputs, train.valid) if ok}
        for name in ("validation", "test"):
            if name in parsed:
                p, rows = parsed[name], np.flatnonzero(keep[name])
                leaked = np.array([p.inputs[i] in seen for i in rows], dtype=bool)
                keep[name][rows[leaked]] = False
                report["leaks"][name] = int(leaked.sum())
                seen.update(p.inputs[i] for i in rows[~leaked])

    # Cut what's missing from train: whole groups at a time, each answer getting its share.
    progress("Splitting", 0.0)
    train_rows = np.flatnonzero(keep["train"])
    fractions = {"validation": 0 if "validation" in parsed else opts["val_percent"] / 100,
                 "test": 0 if "test" in parsed else opts["test_percent"] / 100}
    keep_together = opts["keep_together"]
    groups: list = list(range(len(train.inputs)))
    if keep_together.startswith("column:"):
        column = keep_together.split(":", 1)[1]
        values = extra_columns(frames["train"], fmt, only=column).get(column)
        if values is None:
            notes.append(f"Train has no {column} column, so rows with the same {INPUT_WORD[fmt]} were kept together "
                         "instead.")
            keep_together = "input"
        else:  # rows with the same input stay together too, or a state could still be on both sides
            groups = _together([t if ok else None for t, ok in zip(train.inputs, train.valid)], values)
    if keep_together == "input":
        groups = train.inputs
    cells = [answer_cells(label) if ok else [] for label, ok in zip(train.labels, train.valid)] \
        if fmt == "decision" else None
    balance = opts["balance"] and cells is not None and any(fractions.values())
    cut = split_groups(groups, train_rows, fractions["validation"], fractions["test"], opts["seed"],
                       cells if balance else None)
    report["keep_together"], report["balanced"] = keep_together, balance
    for name, fraction in fractions.items():
        wanted, got = _half_up(len(train_rows) * fraction), len(cut[name])
        if wanted and not got:
            notes.append(f"{name.capitalize()} got no rows: the rows kept together form too few groups to share out.")
        elif wanted and got > 1.5 * wanted and got - wanted >= 5:
            notes.append(f"{name.capitalize()} has {got:,} rows rather than {wanted:,}, because the rows kept together "
                         "come in big groups.")

    progress("Saving", 0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = {"train": (frames["train"], cut["train"])}
    answered = {"train": (cells, cut["train"])}
    for name in ("validation", "test"):
        if name in parsed:
            p, rows = parsed[name], np.flatnonzero(keep[name])
            chosen[name] = (frames[name], rows)
            if cells is not None:
                answered[name] = ([answer_cells(label) if ok else [] for label, ok in zip(p.labels, p.valid)], rows)
        else:
            chosen[name] = (frames["train"], cut[name])
            answered[name] = (cells, cut[name])
    for name, (frame, rows) in chosen.items():
        frame.iloc[rows].reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet", index=False)
        report["rows"][name] = int(len(rows))
    if fmt == "decision":  # a decision model trains on (row, question) pairs, so these are what a step's batch counts
        report["items"] = {"train": int(sum(len(_value(q)) for q in frames["train"]["questions"].iloc[cut["train"]]))}
        frame, rows = chosen["validation"] if len(chosen["validation"][1]) else chosen["train"]
        report["questions"] = question_summary(frame["questions"].iloc[rows])
        report["answers"] = answer_summary(answered)
        untrained = [(q, a, sum(c.values())) for q, answers in report["answers"].items() for a, c in answers.items()
                     if not c["train"]]
        if untrained:
            listed = _listed([f"{q} {_said(a)} ({n:,} rows in all)" for q, a, n in untrained[:5]]
                             + ([f"{len(untrained) - 5} more"] if len(untrained) > 5 else []))
            notes.append(f"Train has no rows with {listed}, so the model can't learn "
                         f"{'that answer' if len(untrained) == 1 else 'those answers'}.")
        for name, fraction in fractions.items():  # answers a split cut from train has no rows of
            absent = [(q, a, sum(c.values())) for q, answers in report["answers"].items() for a, c in answers.items()
                      if fraction and len(cut[name]) and not c[name]]
            if absent:
                listed = _listed([f"{q} {_said(a)} ({n:,} rows in all)" for q, a, n in absent[:5]]
                                 + ([f"{len(absent) - 5} more"] if len(absent) > 5 else []))
                notes.append(f"{name.capitalize()} has no rows with {listed}, so its checks can't score "
                             f"{'that answer' if len(absent) == 1 else 'those answers'}.")
    report["given"] = {name: name in parsed for name in ("validation", "test")}
    report["seconds"] = round(time.time() - started, 1)
    (out_dir / "request.json").write_text(json.dumps(
        {"sources": {n: request.get(n) for n in SPLITS}, "options": opts}, indent=1))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    return report


# ---------------------------------------------------------------- the Hugging Face Hub

def search_hub(query: str, token: str | None, limit: int = 12) -> list[dict]:
    from huggingface_hub import HfApi

    from .models_check import PERMISSIVE, _licence

    found = HfApi().list_datasets(search=query or None, sort="downloads", limit=limit, token=token,
                                  expand=["downloads", "gated", "private", "cardData", "tags"])
    out = []
    for d in found:
        tags = list(d.tags or [])
        licence = _licence(d.card_data, tags)
        out.append({
            "id": d.id,
            "downloads": d.downloads or 0,
            "licence": licence,
            "permissive": licence in PERMISSIVE,
            "gated": bool(d.gated),
            "private": bool(d.private),
            "size": next((t.split(":", 1)[1] for t in tags if t.startswith("size_categories:")), None),
        })
    return out


def hub_configs(repo: str, token: str | None) -> list[str]:
    import datasets

    return list(datasets.get_dataset_config_names(repo, token=token))[:50]


def hub_splits(repo: str, config: str | None, token: str | None) -> list[str]:
    import datasets

    return list(datasets.get_dataset_split_names(repo, config, token=token))


# ---------------------------------------------------------------- quick look

def inspect(source: dict, token: str | None) -> dict:
    """Row count, format, a few sample rows, and a check of the first 2,000 rows: in those rows, the columns they can be
    kept together by and, for decision data, how often each question gets each answer."""
    if source.get("kind") == "file":
        path = resolve_file(source["path"])
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            file = pq.ParquetFile(path)
            rows = file.metadata.num_rows
            head = next(file.iter_batches(batch_size=2000)).to_pandas() if rows else pd.DataFrame(
                columns=file.schema_arrow.names)
        else:
            frame = read_file(path)
            rows, head = len(frame), frame.head(2000)
    else:
        frame = read_source(source, token)
        rows, head = len(frame), frame.head(2000)
    fmt = detect_format(head.columns)
    checked = parse(head, fmt) if fmt else None

    def cell(value) -> str:
        text = value if isinstance(value, str) else _canonical(_value(value) if isinstance(value, np.ndarray) else value)
        return text if len(text) <= 240 else text[:240] + "…"

    columns = list(FORMATS[fmt]) if fmt else list(head.columns)[:6]
    labels = [label if ok else "" for label, ok in zip(checked.labels, checked.valid)] if checked else []
    answers = None
    if fmt == "decision":
        found = answer_summary({"train": ([answer_cells(label) for label in labels], range(len(labels)))})
        answers = {name: {a: counts["train"] for a, counts in by_answer.items()} for name, by_answer in found.items()}
    return {
        "rows": int(rows),
        "columns": [str(c) for c in head.columns],
        "format": fmt,
        "format_label": LABELS.get(fmt or "", None),
        "preview": [{c: cell(head[c].iloc[i]) for c in columns} for i in range(min(3, len(head)))],
        "checked": len(head),
        "invalid": checked.invalid if checked else None,
        "examples": checked.examples if checked else [],
        "groups": group_choices(head, fmt, labels) if fmt else [],
        "answers": answers,
    }


# ---------------------------------------------------------------- background jobs

@dataclass
class Job:
    id: str
    fingerprint: str
    state: str = "running"  # running, done or failed
    stage: str = "Waiting"
    progress: float = 0.0
    report: dict | None = None
    error: str | None = None

    def public(self) -> dict:
        return {"id": self.id, "fingerprint": self.fingerprint, "state": self.state, "stage": self.stage,
                "progress": round(self.progress, 3), "report": self.report, "error": self.error}


_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)  # one at a time: preparing is memory-hungry


def _run(job: Job, request: dict, token: str | None) -> None:
    def progress(stage: str, fraction: float) -> None:
        job.stage, job.progress = stage, max(0.0, min(1.0, fraction))

    out_dir = config.settings.prepared / job.fingerprint
    try:
        job.report = prepare(request, token, out_dir, progress)
        job.state = "done"
    except Exception as error:  # shown on the screen as is, so the reasons above are written for people
        job.state, job.error = "failed", str(error) or error.__class__.__name__


def question_summary(questions) -> dict:
    """{name: {type, rows}} for each question asked in these rows: what the validation checks will score."""
    found: dict = {}
    for value in questions:
        for name, question in _value(value).items():
            found.setdefault(name, {"type": question["type"], "rows": 0})["rows"] += 1
    return dict(sorted(found.items()))


def cached(fp: str) -> dict | None:
    folder = config.settings.prepared / fp
    path = folder / "report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if report.get("format") == "decision" and "questions" not in report:  # prepared before reports listed them
        split = "validation" if report["rows"].get("validation") else "train"
        report["questions"] = question_summary(pd.read_parquet(folder / f"{split}.parquet", columns=["questions"])
                                               ["questions"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=1))
        os.replace(tmp, path)
    return report


def start(request: dict, token: str | None, run: bool = True) -> Job | None:
    """The job preparing this request: an earlier result, one still running, or (if run) a new one."""
    sources = {name: identity(request[name], token) for name in SPLITS if request.get(name)}
    fp = fingerprint(request, sources)
    report = cached(fp)
    if report is not None:
        return Job(id=fp, fingerprint=fp, state="done", stage="Done", progress=1.0, report=report)
    with _lock:
        running = next((j for j in _jobs.values() if j.fingerprint == fp and j.state == "running"), None)
        if running or not run:
            return running
        job = Job(id=uuid.uuid4().hex[:12], fingerprint=fp)
        _jobs[job.id] = job
    _pool.submit(_run, job, request, token)
    return job


def job(job_id: str) -> Job | None:
    return _jobs.get(job_id)
