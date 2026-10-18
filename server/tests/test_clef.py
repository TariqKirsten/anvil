"""The Clef way of training a decision model: Cloudflare's head on a tiny model, end to end on the CPU."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from anvil import tasks
from anvil.vendor.clef import joint_schema_model as clef
from test_training import (SERVER, decision_rows, finish, make_run, metrics, prepared, run_config, start,
                           tiny_model)  # noqa: F401  (tiny_model is a fixture)

VENDOR = SERVER / "anvil" / "vendor" / "clef"


def clef_config(model, data, **hyper):
    cfg = run_config(model, data, maxLength=1024, **hyper)
    cfg["method"] = "clef"
    return cfg


def test_the_vendored_clef_files_are_cloudflares_own():
    digests = {"joint_schema_model.py": "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3",
               "LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"}
    for name, digest in digests.items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name


def test_answers_map_onto_clefs_options():
    noul = {"type": "noul", "instructions": "Rude?"}
    assert tasks.clef_target(noul, True, ("true", "false")) == [1.0, 0.0]
    assert tasks.clef_target(noul, {"label": False, "probabilities": {"true": 0.2, "false": 0.8}},
                             ("true", "false")) == pytest.approx([0.2, 0.8])
    choice = {"type": "choice", "criteria": {"spam": "ads", "fine": "normal"}}
    ids = tuple(o for o, _ in clef.question_options(choice))
    assert ids == ("fine", "spam") and tasks.clef_target(choice, "spam", ids) == [0.0, 1.0]
    score = {"type": "score", "criteria": ["none", "some", "lots"]}
    assert tasks.clef_target(score, 2, ("0", "1", "2")) == [0.0, 0.0, 1.0]


def test_a_new_head_is_sized_like_cloudflares():
    assert tasks.clef_head_config(2048) == {"hidden_size": 2048, "width": 512, "routing_layers": 2, "layers": 4,
                                            "heads": 8, "feedforward": 2048, "dropout": 0.0}
    assert tasks.clef_head_config(4096)["width"] == 1024 and tasks.clef_head_config(4096)["heads"] == 16


def test_a_clef_run_trains_its_head_and_saves_it_like_a_clef_release(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    run_dir = make_run(tmp_path, clef_config(tiny_model, data, maxSteps=6, every=3))
    status = finish(start(run_dir), run_dir)
    assert status["state"] == "done" and status["head_m"] > 0
    evals = [m for m in metrics(run_dir) if m["type"] == "eval"]
    assert [m["step"] for m in evals] == [0, 3, 6] and {"auroc", "accuracy", "catch_rate", "ece"} <= set(evals[-1])
    assert all(np.isfinite(m["loss"]) for m in metrics(run_dir) if m["type"] == "train")
    assert (run_dir / "checkpoints" / "step-0000006" / "head" / "joint_head.safetensors").exists()
    merged = run_dir / "model" / "merged"
    config = json.loads((merged / "joint_head_config.json").read_text())
    assert config == tasks.clef_head_config(32)
    assert json.loads((merged / "anvil.json").read_text())["method"] == "clef"

    # Cloudflare's own code loads the result and answers a SystemOne request with it.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    head = clef.JointSchemaHead(**config)
    head.load_state_dict(load_file(str(merged / "joint_head.safetensors")), strict=True)
    model = clef.ClefModel(AutoModelForCausalLM.from_pretrained(merged), head).eval()
    tok = AutoTokenizer.from_pretrained(merged)
    request = {"model": "tiny", "state": {"message": "rude the spam no"},
               "questions": {"rude": {"type": "noul", "instructions": "Is it rude?"},
                             "kind": {"type": "choice", "instructions": "Which kind?",
                                      "criteria": {"spam": "ads", "fine": "normal"}}}}
    answer = clef.systemone(model, SimpleNamespace(tokenizer=tok), request, max_length=1024)
    assert 0 <= answer["answers"]["rude"]["noul"] <= 1 and answer["answers"]["kind"]["choice"] in ("spam", "fine")

    # ...and gives the same scores as anvil's own path through the head.
    task = tasks.DecisionClef(tok, 1024, head, config)
    parsed = {"state": [request["state"]], "questions": [request["questions"]], "gold": [{"rude": True, "kind": "spam"}]}
    with torch.no_grad():
        ours = task.logits(model.language_model, task.collate([0], parsed))[0]
        theirs = model(clef.collate_records([clef.encode_record(tok, request, max_length=1024)], tok.pad_token_id,
                                            torch.device("cpu")))[0]
    for a, b in zip(ours, theirs):
        assert torch.allclose(a, b, atol=1e-5)


def test_a_new_head_trains_alone_first(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(80), decision_rows(12, seed=1))
    cfg = clef_config(tiny_model, data, maxSteps=20, every=2)
    cfg["checkpoints"].update(keepBest=100, patience=0)
    run_dir = make_run(tmp_path, cfg)
    status = finish(start(run_dir), run_dir)
    assert status["head_first"] == 2  # a tenth of 20 steps

    def lora_b(step):
        weights = load_file(str(run_dir / "checkpoints" / f"step-{step:07d}" / "adapter" / "adapter_model.safetensors"))
        return max(float(v.abs().max()) for k, v in weights.items() if "lora_B" in k)

    def head(step):
        return load_file(str(run_dir / "checkpoints" / f"step-{step:07d}" / "head" / "joint_head.safetensors"))

    assert lora_b(2) == 0.0 and lora_b(4) > 0.0  # LoRA's B starts at zero and stays there while the head trains alone
    assert any(not torch.equal(v, head(4)[k]) for k, v in head(2).items())


def test_a_clef_run_stops_and_resumes(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(400), decision_rows(8, seed=1))
    run_dir = make_run(tmp_path, clef_config(tiny_model, data, maxSteps=200, every=1000))
    proc = start(run_dir)
    deadline = time.time() + 240
    while time.time() < deadline:
        status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
        if status.get("state") == "running" and status.get("step", 0) >= 3:
            break
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    stopped = finish(proc, run_dir)
    assert stopped["state"] == "stopped", stopped
    assert (run_dir / "checkpoints" / f"step-{stopped['step']:07d}" / "head" / "joint_head.safetensors").exists()
    done = finish(start(run_dir, resume=True), run_dir, timeout=400)
    assert done["state"] == "done" and done["step"] == 200


def test_the_judge_checks_clef_checkpoints_with_their_heads(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    cfg = clef_config(tiny_model, data, maxSteps=6, every=2, learningRate="3e-2")
    cfg["judge_gpu"] = 0
    run_dir = make_run(tmp_path, cfg)
    (run_dir / "judge").mkdir()
    env = {**os.environ, "ANVIL_DEVICE": "cpu", "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(SERVER)}
    judge = subprocess.Popen([sys.executable, "-m", "anvil.judge", str(run_dir)], env=env,
                             stdout=(run_dir / "judge" / "log.txt").open("ab"), stderr=subprocess.STDOUT)
    status = finish(start(run_dir), run_dir)
    assert judge.wait(timeout=240) == 0, (run_dir / "judge" / "log.txt").read_text()[-3000:]
    by_judge = {m["step"]: m["val_loss"] for m in metrics(run_dir) if m["type"] == "eval" and m.get("by") == "judge"}
    assert sorted(by_judge) == [0, 2, 4, 6] and len(set(by_judge.values())) == 4  # each scored with its own head
    assert status["final_test"]["step"] == status["best_step"]


def test_two_cards_learn_the_clef_way_like_one_card_with_the_whole_batch(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(60), decision_rows(12, seed=1))
    one = make_run(tmp_path, clef_config(tiny_model, data, batchSize=4, maxSteps=4, every=4), "one")
    two = make_run(tmp_path, clef_config(tiny_model, data, batchSize=2, maxSteps=4, every=4), "two")
    finish(start(one), one)
    finish(start(two, cards=2), two)
    a = [m["loss"] for m in metrics(one) if m["type"] == "train"]
    b = [m["loss"] for m in metrics(two) if m["type"] == "train"]
    assert len(a) == len(b) == 4 and np.allclose(a, b, rtol=1e-3, atol=1e-5), (a, b)


def test_too_short_a_max_length_says_so_before_training(tiny_model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model)
    head = clef.JointSchemaHead(**tasks.clef_head_config(32))
    task = tasks.DecisionClef(tok, 64, head, tasks.clef_head_config(32))
    with pytest.raises(ValueError, match="Max length 64 is too short for the Clef prompt"):
        task.items(decision_rows(3))
