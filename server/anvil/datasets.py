"""Datasets: reading them, checking every row, removing duplicates, and cutting validation and test sets from train.

A dataset's format is recognised from its columns:
- decision: state, questions, gold (System One style, as JSON strings or objects)
- chat: messages, a list of {role, content}
- pairs: prompt, completion
- text: text

Every row is reduced to three parts, so one set of rules covers every format:
- input: what the model reads (the state, or the user's side of a chat). Copies and leaks are found on this.
- output: what is asked or written (the questions, or the assistant's side).
- label: the answer, for decision rows.

Two rows are copies when their inputs match and their outputs match; when the answers differ too, the copy is also a
conflict. Exact copies compare normalised text. Near-copies use MinHash over word trigrams, the guard pipeline's rule:
inputs that share at least 70% of their 3-word chunks. Inputs that are copies or near-copies of each other form a group,
and a group always lands in one split, so the test never holds a state the model trained on.
"""

from __future__ import annotations

import hashlib
import json
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
NUM_PERM = 64
SIMILAR = 0.7
VERSION = 1  # bump when the preparation rules change, so data prepared under the old rules isn't reused
SPLITS = ("train", "validation", "test")

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

    The rules are the ones Unsloth Studio and System One apply."""
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


# ---------------------------------------------------------------- copies, near-copies and groups

def _shingles(text: str) -> set[bytes]:
    words = text.split()
    return {" ".join(words[i:i + 3]).encode() for i in range(max(1, len(words) - 2))}


class Inputs:
    """Every distinct input of a dataset, with its group: inputs that are copies or near-copies of each other (directly
    or through a chain) share a group. Also answers whether another input copies one of these."""

    def __init__(self, texts: list[str], near: bool, progress: Progress = _quiet):
        self.unique = list(dict.fromkeys(texts))
        self.index = {t: i for i, t in enumerate(self.unique)}
        self.near = near
        parent = list(range(len(self.unique)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        self.hashes, self.lsh = [], None
        if near and self.unique:
            from datasketch import MinHash, MinHashLSH

            progress("Finding near-copies", 0.0)
            self.hashes = []
            step = 20_000
            for start in range(0, len(self.unique), step):
                chunk = self.unique[start:start + step]
                self.hashes += MinHash.bulk([_shingles(t) for t in chunk], num_perm=NUM_PERM)
                progress("Finding near-copies", 0.5 * min(1.0, (start + step) / len(self.unique)))
            self.lsh = MinHashLSH(threshold=SIMILAR, num_perm=NUM_PERM)
            for i, h in enumerate(self.hashes):
                for j in self.lsh.query(h):
                    if self.hashes[j].jaccard(h) >= SIMILAR:
                        parent[find(i)] = find(j)
                self.lsh.insert(i, h)
                if i % 20_000 == 0:
                    progress("Finding near-copies", 0.5 + 0.5 * i / len(self.unique))
        self.group = [find(i) for i in range(len(self.unique))]

    def group_of(self, text: str) -> int:
        return self.group[self.index[text]]

    def copied(self, texts: list[str]) -> np.ndarray:
        """True where a text is a copy or near-copy of one of these inputs."""
        hit = np.array([t in self.index for t in texts], dtype=bool)
        if self.near and self.lsh is not None:
            from datasketch import MinHash

            todo = [i for i, t in enumerate(texts) if not hit[i]]
            hashes = MinHash.bulk([_shingles(texts[i]) for i in todo], num_perm=NUM_PERM)
            for i, h in zip(todo, hashes):
                hit[i] = any(self.hashes[j].jaccard(h) >= SIMILAR for j in self.lsh.query(h))
        return hit


def drop_copies(p: Parsed, inputs: Inputs | None) -> tuple[np.ndarray, dict]:
    """Rows to keep: the first of every set of copies. With inputs given, near-copies count as copies too."""
    keep = p.valid.copy()
    first_exact: dict = {}
    first_near: dict = {}
    counts = {"exact": 0, "near": 0, "conflicts": 0}
    for i in np.flatnonzero(p.valid):
        exact = (p.inputs[i], p.outputs[i])
        if exact in first_exact:
            keep[i] = False
            counts["exact"] += 1
            counts["conflicts"] += first_exact[exact] != p.labels[i]
            continue
        first_exact[exact] = p.labels[i]
        if inputs is not None and inputs.near:
            near = (inputs.group_of(p.inputs[i]), p.outputs[i])
            if near in first_near:
                keep[i] = False
                counts["near"] += 1
                counts["conflicts"] += first_near[near] != p.labels[i]
                continue
            first_near[near] = p.labels[i]
    return keep, counts


def split_groups(groups: list, rows: np.ndarray, val_fraction: float, test_fraction: float,
                 seed: int) -> dict[str, np.ndarray]:
    """Cut validation and test from rows, whole groups at a time, in a shuffled but repeatable order."""
    members: dict = {}
    for row in rows:
        members.setdefault(groups[row], []).append(row)
    keys = list(members)
    order = np.random.default_rng(seed).permutation(len(keys))
    want = {"test": round(len(rows) * test_fraction), "validation": round(len(rows) * val_fraction)}
    out: dict[str, list] = {"train": [], "validation": [], "test": []}
    for k in order:
        group = members[keys[k]]
        target = next((s for s in ("test", "validation") if len(out[s]) < want[s]), "train")
        out[target] += group
    return {name: np.array(sorted(r), dtype=np.int64) for name, r in out.items()}


# ---------------------------------------------------------------- preparing a dataset

def options(request: dict) -> dict:
    """The settings that change the result, with fixed types: 5 and 5.0 must give the same fingerprint."""
    return {"dedupe": bool(request.get("dedupe", True)), "near_copies": bool(request.get("near_copies", True)),
            "val_percent": float(request.get("val_percent", 5)), "test_percent": float(request.get("test_percent", 5)),
            "keep_together": str(request.get("keep_together", "input")), "seed": int(request.get("seed", 3407))}


def fingerprint(request: dict, identities: dict) -> str:
    blob = _canonical({"version": VERSION, "sources": identities, "options": options(request)})
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def prepare(request: dict, token: str | None, out_dir: Path, progress: Progress = _quiet) -> dict:
    """Check, deduplicate and split. Writes train/validation/test.parquet and report.json into out_dir."""
    started = time.time()
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

    dedupe = bool(request.get("dedupe", True))
    near = dedupe and bool(request.get("near_copies", True))
    keep_together = request.get("keep_together", "input")
    notes = []

    train = parsed["train"]
    # Groups need near-copies whenever rows that share an input must stay together, even if near-copies are kept.
    want_groups = keep_together == "input" and not (request.get("validation") and request.get("test"))
    inputs = Inputs([t for t, ok in zip(train.inputs, train.valid) if ok], near or want_groups, progress)

    report: dict = {"format": fmt, "format_label": LABELS[fmt], "invalid": {}, "examples": {}, "removed": {},
                    "leaks": {}, "rows": {}, "notes": notes}
    keep = {}
    for name, p in parsed.items():
        report["invalid"][name] = p.invalid
        report["examples"][name] = p.examples
        if dedupe:
            own = inputs if name == "train" else (Inputs([t for t, ok in zip(p.inputs, p.valid) if ok], near)
                                                  if near else None)
            keep[name], report["removed"][name] = drop_copies(p, own if near else None)
        else:
            keep[name], report["removed"][name] = p.valid.copy(), {"exact": 0, "near": 0, "conflicts": 0}

    # Validation and test files you gave: drop rows whose input is (a near-copy of) a train input or, for test,
    # a validation input.
    for name in ("validation", "test"):
        if name in parsed:
            rows = np.flatnonzero(keep[name])
            leaked = inputs.copied([parsed[name].inputs[i] for i in rows]) if dedupe else np.zeros(len(rows), bool)
            if name == "test" and "validation" in parsed and dedupe:
                val_rows = np.flatnonzero(keep["validation"])
                val_inputs = Inputs([parsed["validation"].inputs[i] for i in val_rows], near)
                leaked |= val_inputs.copied([parsed[name].inputs[i] for i in rows])
            keep[name][rows[leaked]] = False
            report["leaks"][name] = int(leaked.sum())

    # Cut what's missing from train.
    progress("Splitting", 0.0)
    train_rows = np.flatnonzero(keep["train"])
    val_fraction = 0 if "validation" in parsed else float(request.get("val_percent", 5)) / 100
    test_fraction = 0 if "test" in parsed else float(request.get("test_percent", 5)) / 100
    if keep_together == "source" and "source" not in frames["train"].columns:
        notes.append("Train has no source column, so rows were kept together by their input instead.")
        keep_together = "input"
    if keep_together == "source":
        groups = frames["train"]["source"].astype(str).tolist()
    elif keep_together == "input":
        groups = [inputs.group_of(t) if ok else -1 for t, ok in zip(train.inputs, train.valid)]
    else:
        groups = list(range(len(train.inputs)))
    cut = split_groups(groups, train_rows, val_fraction, test_fraction, int(request.get("seed", 3407)))

    progress("Saving", 0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = {"train": (frames["train"], cut["train"])}
    for name in ("validation", "test"):
        if name in parsed:
            chosen[name] = (frames[name], np.flatnonzero(keep[name]))
        else:
            chosen[name] = (frames["train"], cut[name])
    for name, (frame, rows) in chosen.items():
        frame.iloc[rows].reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet", index=False)
        report["rows"][name] = int(len(rows))
    if fmt == "decision":  # a decision model trains on (row, question) pairs, so these are what a step's batch counts
        report["items"] = {"train": int(sum(len(_value(q)) for q in frames["train"]["questions"].iloc[cut["train"]]))}
        frame, rows = chosen["validation"] if len(chosen["validation"][1]) else chosen["train"]
        report["questions"] = question_summary(frame["questions"].iloc[rows])
    report["given"] = {name: name in parsed for name in ("validation", "test")}
    report["seconds"] = round(time.time() - started, 1)
    (out_dir / "request.json").write_text(json.dumps(
        {"sources": {n: request.get(n) for n in SPLITS}, "options": options(request)}, indent=1))
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
    """Row count, format, a few sample rows, and a check of the first 2,000 rows."""
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
    return {
        "rows": int(rows),
        "columns": [str(c) for c in head.columns],
        "format": fmt,
        "format_label": LABELS.get(fmt or "", None),
        "preview": [{c: cell(head[c].iloc[i]) for c in columns} for i in range(min(3, len(head)))],
        "checked": len(head),
        "invalid": checked.invalid if checked else None,
        "examples": checked.examples if checked else [],
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
