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
                  "dedupe": True, "keep_together": "input", "seed": 1}, data_settings)
    assert report["format"] == "decision"
    assert report["invalid"]["train"] == 1
    assert report["removed"]["train"] == {"exact": 1, "conflicts": 0}  # the near-copy stays: it's a row of its own
    assert sum(report["rows"].values()) == 121
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
    for split in datasets.SPLITS:  # every row's answers, counted where the row went
        frame = pd.read_parquet(out / f"{split}.parquet")
        golds = [json.loads(g) for g in frame["gold"]]
        assert report["answers"]["bad"]["true"][split] == sum(g.get("bad") is True for g in golds)
        assert report["answers"]["kind"]["a"][split] == sum("kind" in g for g in golds)


def test_data_prepared_before_reports_listed_questions_gets_the_list(data_settings):
    folder = data_settings.prepared / "old"
    folder.mkdir(parents=True)
    pd.DataFrame([row("hi"), row("ho", CHOICE, {"kind": "a"}), row("hu")]).to_parquet(folder / "validation.parquet")
    (folder / "report.json").write_text(json.dumps({"format": "decision", "rows": {"train": 9, "validation": 3}}))
    assert datasets.cached("old")["questions"] == {"bad": {"type": "noul", "rows": 2},
                                                  "kind": {"type": "choice", "rows": 1}}
    assert "questions" in json.loads((folder / "report.json").read_text())  # saved, so it's worked out once


def test_the_same_seed_gives_the_same_split():
    groups = [i // 2 for i in range(100)]
    rows = pd.Series(range(100)).to_numpy()
    answers = [[("q", "yes" if i % 3 else "no")] for i in range(100)]
    for given in (None, answers):
        a = datasets.split_groups(groups, rows, 0.1, 0.1, seed=7, answers=given)
        b = datasets.split_groups(groups, rows, 0.1, 0.1, seed=7, answers=given)
        c = datasets.split_groups(groups, rows, 0.1, 0.1, seed=8, answers=given)
        assert all((a[s] == b[s]).all() for s in a) and any(len(a[s]) != len(c[s]) or (a[s] != c[s]).any() for s in a)
        assert sorted(sum((list(a[s]) for s in a), [])) == list(range(100))  # every row once


def answered(answer, n, start=0, question="intent"):
    """n rows with this answer to a choice question, each with a state of its own."""
    q = {question: {"type": "choice", "instructions": "Which?", "criteria": {k: k for k in "abcd"}}}
    return [row(f"message {start + i} for {answer}", q, {question: answer}) for i in range(n)]


def counts(out, split, question="intent"):
    frame = pd.read_parquet(out / f"{split}.parquet")
    found = {}
    for gold in frame["gold"]:
        answer = json.loads(gold).get(question)
        if answer is not None:
            found[answer] = found.get(answer, 0) + 1
    return found


def test_every_answer_gets_its_share_of_validation_and_test(data_settings):
    path = data_settings.dataset_roots[0] / "train.jsonl"
    rows = answered("a", 100) + answered("b", 60) + answered("c", 30) + answered("d", 10)
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    report = run({"train": {"kind": "file", "path": str(path)}, "val_percent": 10, "test_percent": 10, "seed": 5},
                 data_settings)
    out = data_settings.prepared / "out"
    assert report["balanced"] and report["rows"] == {"train": 160, "validation": 20, "test": 20}
    for split in ("validation", "test"):
        assert counts(out, split) == {"a": 10, "b": 6, "c": 3, "d": 1}
        assert {a: c[split] for a, c in report["answers"]["intent"].items()} == {"a": 10, "b": 6, "c": 3, "d": 1}
    assert list(report["answers"]["intent"]) == ["a", "b", "c", "d"]  # most common first


def test_rows_after_the_same_question_are_not_copies(data_settings):
    """A generated state wraps a short reply in text every row shares. The replies are different rows, whatever the
    wrapping: none is dropped, and each answer still gets its share."""
    path = data_settings.dataset_roots[0] / "train.jsonl"
    q = {"intent": {"type": "choice", "instructions": "Which?", "criteria": {"Yes": "y", "No": "n"}}}
    replies = {"Yes": ["yup", "ya", "go ahead", "sure", "ok do it"], "No": ["nah", "nope", "not now", "skip it", "no"]}
    asks = [f"Send the {thing} now?" for thing in ("letter", "invoice", "quote", "report")]
    rows = [row(f"Asked: {ask} | User typed: {reply}", q, {"intent": answer})
            for ask in asks for answer, typed in replies.items() for reply in typed]
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    report = run({"train": {"kind": "file", "path": str(path)}, "val_percent": 10, "test_percent": 10}, data_settings)
    assert report["removed"]["train"] == {"exact": 0, "conflicts": 0} and sum(report["rows"].values()) == 40
    assert counts(data_settings.prepared / "out", "validation") == {"Yes": 2, "No": 2}


def test_rows_answering_several_questions_share_each_answer(data_settings):
    path = data_settings.dataset_roots[0] / "train.jsonl"
    both = {**NOUL, **CHOICE}
    rows = [row(f"text {i}", both, {"bad": i % 4 == 0, "kind": None if i % 5 == 0 else "ab"[i % 2]})
            for i in range(400)]
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    report = run({"train": {"kind": "file", "path": str(path)}, "val_percent": 10, "test_percent": 10, "seed": 2},
                 data_settings)
    for question, answers in report["answers"].items():
        for answer, where in answers.items():
            total = sum(where.values())
            for split in ("validation", "test"):  # within a row of its share, though rows carry both questions
                assert abs(where[split] - total / 10) <= 1.5, (question, answer, where)


def test_an_answer_too_rare_to_share_stays_in_train_and_says_so(data_settings):
    path = data_settings.dataset_roots[0] / "train.jsonl"
    pd.DataFrame(answered("a", 200) + answered("d", 3)).to_json(path, orient="records", lines=True)
    report = run({"train": {"kind": "file", "path": str(path)}, "val_percent": 5, "test_percent": 5}, data_settings)
    assert report["answers"]["intent"]["d"] == {"train": 3, "validation": 0, "test": 0}
    assert any(n.startswith("Validation has no rows with intent d (3 rows in all)") for n in report["notes"])


def test_rows_can_be_kept_together_by_a_column_or_a_key_of_meta(data_settings):
    path = data_settings.dataset_roots[0] / "train.jsonl"
    rows = []
    for i in range(300):
        r = row(f"reply {i}", NOUL, {"bad": (i // 12) % 2 == 0})  # every topic and question gets both answers
        r["topic"] = f"topic {i % 12}"
        r["meta"] = json.dumps({"ask": f"question {i % 15}", "style": "short"})
        rows.append(r)
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    for column, key in (("topic", "topic"), ("ask", "meta")):
        report = run({"train": {"kind": "file", "path": str(path)}, "val_percent": 10, "test_percent": 10,
                      "keep_together": f"column:{column}"}, data_settings, name=column)
        assert report["keep_together"] == f"column:{column}" and not report["notes"]
        where: dict = {}
        for split in datasets.SPLITS:
            frame = pd.read_parquet(data_settings.prepared / column / f"{split}.parquet")
            values = frame[column] if key == column else [json.loads(m)[column] for m in frame["meta"]]
            for value in values:
                where.setdefault(value, set()).add(split)
        assert all(len(splits) == 1 for splits in where.values())
        assert report["rows"]["validation"] and report["rows"]["test"]


def test_rows_with_the_same_state_stay_together_whatever_their_column_says(data_settings):
    path = data_settings.dataset_roots[0] / "train.jsonl"
    rows = []
    for i in range(200):
        for questions, gold, source in ((NOUL, {"bad": i % 2 == 0}, f"first {i}"), (CHOICE, {"kind": "a"}, f"second {i}")):
            r = row(f"message {i}", questions, gold)  # the same state twice, under two sources
            r["source"] = source
            rows.append(r)
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    report = run({"train": {"kind": "file", "path": str(path)}, "val_percent": 10, "test_percent": 10,
                  "keep_together": "column:source"}, data_settings)
    where: dict = {}
    for split in datasets.SPLITS:
        for state in pd.read_parquet(data_settings.prepared / "out" / f"{split}.parquet")["state"]:
            where.setdefault(state, set()).add(split)
    assert report["rows"]["validation"] and all(len(splits) == 1 for splits in where.values())


def test_a_split_of_few_big_groups_still_gets_one(data_settings):
    """Three sources, each far bigger than validation's 5%: validation and test still get one each, not nothing."""
    path = data_settings.dataset_roots[0] / "train.jsonl"
    rows = []
    for i in range(600):
        r = row(f"message {i}", NOUL, {"bad": i % 4 == 0})
        r["source"] = ["aegis", "wild", "civil"][i % 3]
        rows.append(r)
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    report = run({"train": {"kind": "file", "path": str(path)}, "keep_together": "column:source"}, data_settings)
    assert report["rows"] == {"train": 200, "validation": 200, "test": 200}
    assert "Validation has 200 rows rather than 30, because the rows kept together come in big groups." in report["notes"]


def test_an_answer_train_never_gets_is_said(data_settings):
    """Kept together by a source per answer: train keeps one source, so it lacks an answer, and the notes say so."""
    path = data_settings.dataset_roots[0] / "train.jsonl"
    rows = []
    for i in range(300):
        r = row(f"message {i}", CHOICE, {"kind": "b" if i % 3 == 0 else "a"})
        r["source"] = ["aegis", "wild", "civil"][i % 3]  # every "b" is aegis
        rows.append(r)
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    report = run({"train": {"kind": "file", "path": str(path)}, "keep_together": "column:source", "seed": 1},
                 data_settings)
    # each split gets one whole source, so train holds one answer only: either way the other is named
    absent = [answer for answer, where in report["answers"]["kind"].items() if not where["train"]]
    assert len(absent) == 1
    rows_in_all = 100 if absent[0] == "b" else 200
    assert (f"Train has no rows with kind {absent[0]} ({rows_in_all} rows in all), so the model can't learn that "
            "answer.") in report["notes"]


def test_keeping_together_by_a_column_train_lacks_keeps_states_together(data_settings):
    path = data_settings.dataset_roots[0] / "train.parquet"
    decision_dataset().to_parquet(path)
    report = run({"train": {"kind": "file", "path": str(path)}, "keep_together": "column:ask"}, data_settings)
    assert report["keep_together"] == "input"
    assert report["notes"][0] == "Train has no ask column, so rows with the same state were kept together instead."


def test_keep_together_by_source_means_the_source_column():
    assert datasets.options({"keep_together": "source"})["keep_together"] == "column:source"
    assert datasets.options({})["keep_together"] == "input" and datasets.options({})["balance"] is True


def test_a_test_file_loses_rows_it_shares_with_train(data_settings):
    root = data_settings.dataset_roots[0]
    pd.DataFrame([row(sentence(i)) for i in range(20)]).to_parquet(root / "train.parquet")
    near = sentence(6).replace(f"{WORDS[(6 * 7) % 26]}6", "changed", 1)  # one word of sentence 6 changed: kept
    pd.DataFrame([row(sentence(5)), row(sentence(100)), row(near)]).to_json(
        root / "test.jsonl", orient="records", lines=True)
    report = run({"train": {"kind": "file", "path": str(root / "train.parquet")},
                  "test": {"kind": "file", "path": str(root / "test.jsonl")},
                  "val_percent": 10, "dedupe": True, "keep_together": "input"}, data_settings)
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
    assert info["answers"] == {"bad": {"true": 32, "false": 30}, "kind": {"a": 60}}


def test_inspect_offers_the_columns_rows_can_be_kept_together_by(data_settings):
    path = data_settings.dataset_roots[0] / "made.jsonl"
    rows = []
    for i in range(40):
        label = "yes" if i % 2 else "no"
        r = row(f"Asked: question {i % 5}? | User typed: reply {i}", NOUL, {"bad": label == "yes"})
        r["source"] = f"pipe|label={label}|ask=question {i % 5}"
        r["meta"] = json.dumps({"label": label, "ask": f"question {i % 5}", "id": i, "style": "short",
                                "state": "the generated text"})
        rows.append(r)
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    info = logged_in().post("/api/datasets/inspect", json={"source": {"kind": "file", "path": str(path)}}).json()
    # id differs in every row and style never does, so neither keeps anything apart; meta's state is the state column
    assert info["groups"] == [{"name": "source", "values": 10, "answer": False},
                              {"name": "label", "values": 2, "answer": True},
                              {"name": "ask", "values": 5, "answer": False}]


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


def test_keep_together_takes_input_nothing_or_a_column(data_settings):
    path = data_settings.dataset_roots[0] / "train.parquet"
    decision_dataset().to_parquet(path)
    c = logged_in()
    body = {"train": {"kind": "file", "path": str(path)}}
    for together, status in (("column:ask", 200), ("source", 200), ("nothing", 200), ("column:", 400), ("ask", 400)):
        response = c.post("/api/datasets/prepare", params={"run": "false"}, json={**body, "keep_together": together})
        assert response.status_code == status, together


def test_the_fingerprint_ignores_how_numbers_are_written():
    sources = {"train": {"kind": "file", "path": "/x", "size": 1, "mtime": 2}}
    a = datasets.fingerprint({"val_percent": 5, "test_percent": 5, "seed": 3407}, sources)
    b = datasets.fingerprint({"val_percent": 5.0, "test_percent": 5.0, "seed": 3407.0}, sources)
    assert a == b
    assert a != datasets.fingerprint({"val_percent": 10, "test_percent": 5, "seed": 3407}, sources)


def test_normalize_ignores_case_and_spacing():
    assert normalize("  Hello\n  World ") == "hello world"
