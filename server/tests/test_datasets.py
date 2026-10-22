import json
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from anvil import accounts, config, datasets, main
from anvil.datasets import check_chat, check_decision, detect_format, normalize

NOUL = {"bad": {"type": "noul", "instructions": "Is it bad?"}}
CHOICE = {"kind": {"type": "choice", "instructions": "Which kind?", "criteria": {"a": "first", "b": "second"}}}
SCORE = {"level": {"type": "score", "instructions": "How bad?", "criteria": ["mild", "harsh", "severe"]}}
WORDS = ("apple banana cherry delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
         "romeo sierra tango uniform victor whiskey xray yankee zulu").split()


def row(message, questions=NOUL, gold=None):
    gold = {"bad": False} if gold is None else gold
    return {"state": json.dumps({"message": message}), "questions": json.dumps(questions), "gold": json.dumps(gold)}


def sentence(i: int) -> str:
    """Distinct sentences that share no 3-word chunk with each other."""
    return " ".join(f"{WORDS[(i * 7 + k) % 26]}{i}" for k in range(20))


@pytest.fixture
def data_settings(tmp_path, monkeypatch):
    roots = tmp_path / "datasets"
    roots.mkdir()
    s = config.Settings(password="correct horse", data=tmp_path / "data", dataset_roots=(roots,), hf_token=None)
    monkeypatch.setattr(config, "settings", s)
    accounts.ensure_default_user()
    s.uploads.mkdir(parents=True)
    return s


def logged_in():
    c = TestClient(main.app)
    c.post("/api/login", json={"username": "admin", "password": "correct horse"})
    return c


# ---------------------------------------------------------------- formats and row checks

def test_formats_are_recognised_by_their_columns():
    assert detect_format(["state", "questions", "gold", "source"]) == "decision"
    assert detect_format(["messages"]) == "chat"
    assert detect_format(["prompt", "completion"]) == "pairs"
    assert detect_format(["text", "id"]) == "text"
    assert detect_format(["question", "answer"]) is None


def test_good_decision_rows_pass():
    assert check_decision(*row("hi").values())[0] == "hi"
    check_decision(*row("hi", CHOICE, {"kind": "b"}).values())
    check_decision(*row("hi", SCORE, {"level": 2}).values())
    check_decision(*row("hi", NOUL, {"bad": {"label": True, "probabilities": {"true": 0.9, "false": 0.1}}}).values())
    check_decision({"message": "objects work too"}, NOUL, {"bad": True})
    check_decision(*row("asked, not scored", {**NOUL, **CHOICE}, {"bad": True, "kind": None}).values())


@pytest.mark.parametrize("questions, gold, reason", [
    ({**NOUL, **CHOICE}, {"bad": None, "kind": None}, "at least one question needs an answer"),
    ({"bad": {"type": "yesno"}}, {"bad": True}, "needs the type"),
    ({"bad": {"type": "noul", "hint": "x"}}, {"bad": True}, "may only have"),
    (NOUL, {}, "has no gold answer"),
    (NOUL, {"bad": "maybe"}, "true or false"),
    (CHOICE, {"kind": "c"}, "isn't one of its options"),
    (SCORE, {"level": 3}, "from 0 to 2"),
    (SCORE, {"level": True}, "level number"),
    ({"level": {"type": "score", "criteria": {"a": "x"}}}, {"level": 0}, "must list 1 to 10 levels"),
])
def test_broken_decision_rows_say_why(questions, gold, reason):
    with pytest.raises(ValueError, match=reason):
        check_decision(json.dumps({"message": "hi"}), json.dumps(questions), json.dumps(gold))


def test_chat_rows_need_an_assistant_turn():
    inp, out, _ = check_chat([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])
    assert (inp, out) == ("Hi", "Hello")
    with pytest.raises(ValueError, match="no assistant"):
        check_chat([{"role": "user", "content": "Hi"}])


# ---------------------------------------------------------------- preparing

def decision_dataset(n_states=60):
    rows = []
    for i in range(n_states):
        rows.append(row(sentence(i), NOUL, {"bad": i % 2 == 0}))
        rows.append(row(sentence(i), CHOICE, {"kind": "a"}))
    rows.append(row(sentence(0), NOUL, {"bad": True}))  # an exact copy of row 0
    near = sentence(1).replace(f"{WORDS[7 % 26]}1", "changed", 1)  # sentence 1 with one word changed
    rows.append(row(near, NOUL, {"bad": True}))  # a near-copy of row 2 with a different answer: a conflict
    rows.append({"state": "{not json", "questions": json.dumps(NOUL), "gold": "{}"})
    return pd.DataFrame(rows)


def run(request, settings, name="out"):
    return datasets.prepare(request, None, settings.prepared / name)


def test_prepare_removes_copies_and_keeps_states_on_one_side(data_settings):
    path = data_settings.dataset_roots[0] / "train.parquet"
    decision_dataset().to_parquet(path)
    report = run({"train": {"kind": "file", "path": str(path)}, "val_percent": 10, "test_percent": 10,
                  "dedupe": True, "near_copies": True, "keep_together": "input", "seed": 1}, data_settings)
    assert report["format"] == "decision"
    assert report["invalid"]["train"] == 1
    assert report["removed"]["train"] == {"exact": 1, "near": 1, "conflicts": 1}
    assert sum(report["rows"].values()) == 120
    out = data_settings.prepared / "out"
    states = {s: set(pd.read_parquet(out / f"{s}.parquet")["state"]) for s in datasets.SPLITS}
    assert not states["train"] & states["test"] and not states["train"] & states["validation"]
    assert report["rows"]["test"] >= 10 and report["rows"]["validation"] >= 10
    train = pd.read_parquet(out / "train.parquet")
    assert report["items"]["train"] == sum(len(json.loads(q)) for q in train["questions"])  # questions, not rows
    validation = pd.read_parquet(out / "validation.parquet")
    asked = [name for q in validation["questions"] for name in json.loads(q)]
    assert report["questions"] == {"bad": {"type": "noul", "rows": asked.count("bad")},
                                   "kind": {"type": "choice", "rows": asked.count("kind")}}


def test_data_prepared_before_reports_listed_questions_gets_the_list(data_settings):
    folder = data_settings.prepared / "old"
    folder.mkdir(parents=True)
    pd.DataFrame([row("hi"), row("ho", CHOICE, {"kind": "a"}), row("hu")]).to_parquet(folder / "validation.parquet")
    (folder / "report.json").write_text(json.dumps({"format": "decision", "rows": {"train": 9, "validation": 3}}))
    assert datasets.cached("old")["questions"] == {"bad": {"type": "noul", "rows": 2},
                                                  "kind": {"type": "choice", "rows": 1}}
    assert "questions" in json.loads((folder / "report.json").read_text())  # saved, so it's worked out once


def test_near_copies_stay_together_even_when_kept(data_settings):
    path = data_settings.dataset_roots[0] / "train.parquet"
    decision_dataset().to_parquet(path)
    report = run({"train": {"kind": "file", "path": str(path)}, "val_percent": 20, "test_percent": 20,
                  "dedupe": True, "near_copies": False, "keep_together": "input", "seed": 3}, data_settings)
    assert report["removed"]["train"] == {"exact": 1, "near": 0, "conflicts": 0}
    out = data_settings.prepared / "out"
    where = {}
    for split in datasets.SPLITS:
        for state in pd.read_parquet(out / f"{split}.parquet")["state"]:
            where.setdefault(json.loads(state)["message"].split()[1], set()).add(split)
    # sentence 1 and its near-copy share every word but the first, so key on the second word
    assert len(where[f"{WORDS[8]}1"]) == 1


def test_the_same_seed_gives_the_same_split():
    groups = [i // 2 for i in range(100)]
    rows = pd.Series(range(100)).to_numpy()
    a = datasets.split_groups(groups, rows, 0.1, 0.1, seed=7)
    b = datasets.split_groups(groups, rows, 0.1, 0.1, seed=7)
    c = datasets.split_groups(groups, rows, 0.1, 0.1, seed=8)
    assert all((a[s] == b[s]).all() for s in a) and any(len(a[s]) != len(c[s]) or (a[s] != c[s]).any() for s in a)


def test_a_test_file_loses_rows_it_shares_with_train(data_settings):
    root = data_settings.dataset_roots[0]
    pd.DataFrame([row(sentence(i)) for i in range(20)]).to_parquet(root / "train.parquet")
    pd.DataFrame([row(sentence(5)), row(sentence(100)), row(sentence(101))]).to_json(
        root / "test.jsonl", orient="records", lines=True)
    report = run({"train": {"kind": "file", "path": str(root / "train.parquet")},
                  "test": {"kind": "file", "path": str(root / "test.jsonl")},
                  "val_percent": 10, "dedupe": True, "near_copies": True, "keep_together": "input"}, data_settings)
    assert report["leaks"]["test"] == 1 and report["rows"]["test"] == 2 and report["given"]["test"]


def test_keeping_duplicates_keeps_them(data_settings):
    path = data_settings.dataset_roots[0] / "train.parquet"
    decision_dataset().to_parquet(path)
    report = run({"train": {"kind": "file", "path": str(path)}, "val_percent": 0, "test_percent": 0,
                  "dedupe": False, "keep_together": "nothing"}, data_settings)
    assert report["rows"]["train"] == 122 and report["removed"]["train"]["exact"] == 0


def test_mixed_formats_are_refused(data_settings):
    root = data_settings.dataset_roots[0]
    decision_dataset().to_parquet(root / "train.parquet")
    pd.DataFrame({"text": ["hello there"]}).to_csv(root / "val.csv", index=False)
    with pytest.raises(ValueError, match="same format"):
        run({"train": {"kind": "file", "path": str(root / "train.parquet")},
             "validation": {"kind": "file", "path": str(root / "val.csv")}}, data_settings)


# ---------------------------------------------------------------- the API

def test_upload_lands_in_uploads_without_overwriting(data_settings):
    c = logged_in()
    first = c.put("/api/datasets/upload", params={"name": "my data.jsonl"}, content=b'{"text": "hi"}\n').json()
    second = c.put("/api/datasets/upload", params={"name": "my data.jsonl"}, content=b'{"text": "yo"}\n').json()
    assert first["name"] == "my_data.jsonl" and second["name"] == "my_data-1.jsonl"
    assert c.put("/api/datasets/upload", params={"name": "notes.txt"}, content=b"x").status_code == 400
    listing = c.get("/api/datasets/browse", params={"path": str(data_settings.uploads)}).json()
    assert [f["name"] for f in listing["files"]] == ["my_data-1.jsonl", "my_data.jsonl"]


def test_browsing_stays_inside_the_dataset_folders(data_settings, tmp_path):
    c = logged_in()
    assert c.get("/api/datasets/browse", params={"path": str(tmp_path)}).status_code == 403
    assert c.post("/api/datasets/inspect", json={"source": {"kind": "file", "path": "/etc/passwd"}}).status_code == 403


def test_inspect_shows_format_and_problems(data_settings):
    path = data_settings.dataset_roots[0] / "train.parquet"
    decision_dataset().to_parquet(path)
    info = logged_in().post("/api/datasets/inspect", json={"source": {"kind": "file", "path": str(path)}}).json()
    assert info["rows"] == 123 and info["format"] == "decision" and info["invalid"] == 1
    assert info["preview"][0]["state"].startswith('{"message"')


def test_prepare_job_runs_then_is_cached(data_settings):
    path = data_settings.dataset_roots[0] / "train.parquet"
    decision_dataset().to_parquet(path)
    c = logged_in()
    body = {"train": {"kind": "file", "path": str(path)}, "val_percent": 10, "test_percent": 10}
    assert c.post("/api/datasets/prepare", params={"run": "false"}, json=body).json()["job"] is None
    job = c.post("/api/datasets/prepare", json=body).json()["job"]
    for _ in range(100):
        job = c.get(f"/api/datasets/jobs/{job['id']}").json()["job"]
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done", job
    again = c.post("/api/datasets/prepare", params={"run": "false"}, json=body).json()["job"]
    assert again["state"] == "done" and again["report"] == job["report"]


def test_the_fingerprint_ignores_how_numbers_are_written():
    sources = {"train": {"kind": "file", "path": "/x", "size": 1, "mtime": 2}}
    a = datasets.fingerprint({"val_percent": 5, "test_percent": 5, "seed": 3407}, sources)
    b = datasets.fingerprint({"val_percent": 5.0, "test_percent": 5.0, "seed": 3407.0}, sources)
    assert a == b
    assert a != datasets.fingerprint({"val_percent": 10, "test_percent": 5, "seed": 3407}, sources)


def test_normalize_ignores_case_and_spacing():
    assert normalize("  Hello\n  World ") == "hello world"
