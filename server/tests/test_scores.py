"""Scoring decision models: every question by its type, whatever the dataset calls them, for both methods."""

import contextlib
import json

import pandas as pd
import pytest
import torch

from anvil import config, runs, tasks
from anvil.vendor.clef.joint_schema_model import JointSchemaHead
from test_training import FORM, run_settings, tiny_model  # noqa: F401  (fixtures)


def answers(**questions) -> dict:
    """Outputs as a task gives them, with made-up pooled numbers (the per-question scores don't use them)."""
    return {"loss": [0.5] * 4, "correct": [True, False] * 2, "confidence": [0.7] * 4,
            "questions": {(name, q.pop("type")): q for name, q in questions.items()}}


def test_each_question_is_scored_by_its_type():
    out = answers(flag={"type": "noul", "yes": [0.9, 0.2, 0.7, 0.8], "gold": [True, False, True, False]},
                  kind={"type": "choice", "picked": ["a", "a", "b", "c"], "gold": ["a", "b", "b", "b"]},
                  level={"type": "score", "picked": [0, 2, 1, 1], "gold": [0, 0, 1, 2], "levels": [3, 3, 3, 3]})
    m = tasks.decision_metrics(out, 4, 0.5)
    flag, kind, level = (m["questions"][name] for name in ("flag", "kind", "level"))
    # 3 of the 4 (yes, no) pairs rank the yes higher; one of the two no cases may be flagged, so both yes are caught
    assert flag == {"type": "noul", "rows": 4, "answers": {"yes": 2, "no": 2}, "auroc": 0.75, "catch_rate": 1.0,
                    "false_alarms": 0.5}
    # F1 per answer, 2·hits / (2·hits + wrongly picked + missed): a 2/3, b 1/2, c (picked once, never right) 0
    assert kind["macro_f1"] == pytest.approx((2 / 3 + 1 / 2 + 0) / 3)
    assert kind["accuracy"] == 0.5 and kind["answers"] == {"b": 3, "a": 1} and kind["rows"] == 4
    assert level == {"type": "score", "rows": 4, "answers": {"0": 2, "1": 1, "2": 1}, "accuracy": 0.5, "mae": 0.75}
    assert (m["auroc"], m["catch_rate"], m["macro_f1"], m["mae"]) == (0.75, 1.0, kind["macro_f1"], 0.75)
    # overall: each question's 0-to-1 quality, averaged; a score's is 1 - MAE / 2, three levels being at most 2 apart
    assert m["overall"] == pytest.approx((0.75 + kind["macro_f1"] + (1 - 0.75 / 2)) / 3)
    assert tasks.decision_metrics(out, 4, 0.0)["questions"]["flag"]["catch_rate"] == 0.5  # no false alarm allowed


def test_a_question_with_one_answer_only_has_no_auroc():
    m = tasks.decision_metrics(answers(flag={"type": "noul", "yes": [0.9, 0.7], "gold": [True, True]}), 2, 0.01)
    assert m["questions"]["flag"] == {"type": "noul", "rows": 2, "answers": {"yes": 2, "no": 0}}
    assert "auroc" not in m and "overall" not in m and m["accuracy"] == 0.5


def test_cards_answers_combine_and_a_name_asked_two_ways_stays_two_questions():
    first, second = {}, {}
    tasks.note(first, "x", "noul", yes=0.9, gold=True)
    tasks.note(second, "x", "choice", picked="p", gold="q")
    tasks.note(second, "x", "noul", yes=0.1, gold=False)
    out = tasks.Decisions.combine([{"loss": [1.0], "correct": [True], "confidence": [0.9], "questions": q}
                                   for q in (first, second)])
    assert out["questions"] == {("x", "noul"): {"yes": [0.9, 0.1], "gold": [True, False]},
                                ("x", "choice"): {"picked": ["p"], "gold": ["q"]}}
    assert set(tasks.decision_metrics(out, 2, 0.01)["questions"]) == {"x (noul)", "x (choice)"}


def mixed_rows(n: int) -> pd.DataFrame:
    """Rows that each ask a yes/no, a choice and a score question."""
    questions = {"rude": {"type": "noul", "instructions": "Is it rude?"},
                 "kind": {"type": "choice", "instructions": "Which kind?", "criteria": {"spam": "ads", "fine": "normal"}},
                 "level": {"type": "score", "instructions": "How rude?", "criteria": ["none", "some", "a lot"]}}
    return pd.DataFrame([{"state": json.dumps({"message": f"{'rude' if i % 2 else 'kind'} hello {i}"}),
                          "questions": json.dumps(questions),
                          "gold": json.dumps({"rude": bool(i % 2), "kind": "spam" if i % 3 == 0 else "fine",
                                              "level": i % 3})} for i in range(n)])


@pytest.mark.parametrize("method", ["letters", "clef"])
def test_both_methods_score_every_question_by_its_type(tiny_model, method):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model)
    model = AutoModelForCausalLM.from_pretrained(tiny_model).eval()
    if method == "clef":
        head_config = tasks.clef_head_config(model.config.hidden_size)
        torch.manual_seed(0)
        task = tasks.DecisionClef(tok, 1024, JointSchemaHead(**head_config), head_config)
    else:
        task = tasks.DecisionLetters(tok, 256, 1)
    task.false_alarm_rate = 0.25
    items, parsed = task.items(mixed_rows(9))
    m = task.evaluate(model, items, parsed, 4, contextlib.nullcontext)
    q = m["questions"]
    assert {name: s["type"] for name, s in q.items()} == {"kind": "choice", "level": "score", "rude": "noul"}
    assert q["rude"]["answers"] == {"yes": 4, "no": 5} and q["kind"]["answers"] == {"fine": 6, "spam": 3}
    assert q["level"]["answers"] == {"0": 3, "1": 3, "2": 3} and m["rows"] == 27
    assert {"auroc", "catch_rate", "false_alarms", "macro_f1", "mae", "overall"} <= set(m)
    assert 0 <= m["overall"] <= 1 and 0 <= q["level"]["mae"] <= 2


def test_pick_best_by_must_suit_the_data(run_settings, monkeypatch):
    class Done:
        state, fingerprint = "done", "abc"
        report = {"format": "decision", "format_label": "Decision", "rows": {"train": 10},
                  "questions": {"rude": {"type": "noul", "rows": 5}, "kind": {"type": "choice", "rows": 5}}}

    monkeypatch.setattr(runs.datasets, "start", lambda request, token, run: Done())

    def create(**checkpoints):
        return runs.create({**FORM, "checkpoints": {"every": 500, **checkpoints}}, {"train": {}}, None)

    with pytest.raises(ValueError, match="Mean absolute error needs score questions"):
        create(pickBy="mae")
    with pytest.raises(ValueError, match="0 to 50%"):
        create(pickBy="auroc", falseAlarmPercent=80)
    run_id = create(pickBy="macro_f1", falseAlarmPercent=0.5)
    saved = json.loads((config.settings.data / "runs" / run_id / "config.json").read_text())["checkpoints"]
    assert saved == {"every": 500, "pickBy": "macro_f1", "falseAlarmPercent": 0.5}
    assert runs._check_picking({"pickBy": "auroc"}, "lm", {})["pickBy"] == "val_loss"  # a language model has one
    assert runs._check_picking({"pickBy": "mae"}, "decision", {})["pickBy"] == "mae"  # data prepared before the list
