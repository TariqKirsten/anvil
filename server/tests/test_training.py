"""Training end to end on the CPU, with a tiny random Llama and a tiny tokenizer made here, plus the pieces it uses."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anvil import config, runs, tasks

SERVER = Path(__file__).resolve().parents[1]
LETTER_LINE = " ".join(f"{c}." for c in tasks.LETTERS) + " " + " ".join(tasks.LETTERS)
WORDS = "the a message question answer yes no is it rude kind spam fine hello world thanks please help".split()


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    corpus = [LETTER_LINE] * 200 + [" ".join(WORDS)] * 50 + ["message: hello\n\nQuestion: Is it rude?\nA. yes\nB. no\nAnswer: A"] * 50
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<pad>", "<|end|>", "<|user|>", "<|assistant|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|end|>", pad_token="<pad>")
    tok.chat_template = ("{% for m in messages %}{{ '<|' + m['role'] + '|>' + m['content'] + '<|end|>' }}{% endfor %}"
                         "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}")
    assert all(len(tok.encode(f" {c}", add_special_tokens=False)) == 1 for c in tasks.LETTERS)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id, tie_word_embeddings=False))
    folder = tmp_path_factory.mktemp("tiny-model")
    model.save_pretrained(folder)
    tok.save_pretrained(folder)
    return folder


def decision_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rude = bool(rng.random() < 0.5)
        words = " ".join(rng.choice(WORDS, 6))
        message = f"{'rude ' if rude else 'kind '}{words}"
        rows.append({"state": json.dumps({"message": message}),
                     "questions": json.dumps({"rude": {"type": "noul", "instructions": "Is it rude?"}}),
                     "gold": json.dumps({"rude": rude})})
        if i % 3 == 0:
            rows.append({"state": json.dumps({"message": message}),
                         "questions": json.dumps({"kind": {"type": "choice", "instructions": "Which kind?",
                                                           "criteria": {"spam": "ads", "fine": "normal"}}}),
                         "gold": json.dumps({"kind": "fine"})})
    return pd.DataFrame(rows)


def prepared(tmp: Path, train: pd.DataFrame, val: pd.DataFrame) -> Path:
    folder = tmp / "prepared"
    folder.mkdir(parents=True, exist_ok=True)
    train.to_parquet(folder / "train.parquet")
    val.to_parquet(folder / "validation.parquet")
    val.to_parquet(folder / "test.parquet")
    return folder


def run_config(model: Path, data: Path, train_as="decision", fmt="decision", every=3, **hyper) -> dict:
    h = {"learningRate": "3e-3", "epochs": 1, "batchSize": 4, "loraRank": 4, "loraAlpha": 8, "warmupPercent": 0,
         "scheduler": "cosine", "weightDecay": 0.0, "gradAccum": 1, "maxLength": 128, "seed": 1, "precision": "bf16",
         "gradCheckpointing": False, "maxSteps": 0, **hyper}
    return {"id": "test", "name": "test", "train_as": train_as, "method": "letters" if train_as == "decision" else fmt,
            "training_type": "lora",
            "model": {"source": "folder", "id": str(model), "kind": "model", "base_model": None, "custom_code": False},
            "data": {"prepared": str(data), "format": fmt, "rows": {}}, "gpu": 0, "hyper": h,
            "checkpoints": {"every": every, "keepBest": 3, "patience": 3, "pickBy": "auroc"}, "created": time.time()}


def start(run_dir: Path, resume=False, cards=1) -> subprocess.Popen:
    """The trainer, as the server starts it; several "cards" are several CPU processes (gloo instead of RCCL)."""
    env = {**os.environ, "ANVIL_DEVICE": "cpu", "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(SERVER), "OMP_NUM_THREADS": "2"}
    log = (run_dir / "log.txt").open("ab")
    return subprocess.Popen(runs.command("anvil.train", run_dir, cards, ["--resume"] if resume else []),
                            env=env, stdout=log, stderr=subprocess.STDOUT)


def make_run(tmp: Path, cfg: dict, name: str = "run") -> Path:
    run_dir = tmp / name
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(cfg))
    return run_dir


def finish(proc: subprocess.Popen, run_dir: Path, timeout=240) -> dict:
    assert proc.wait(timeout=timeout) == 0, (run_dir / "log.txt").read_text()[-3000:]
    return json.loads((run_dir / "status.json").read_text())


def metrics(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------- end to end

def test_a_decision_run_trains_checkpoints_and_saves(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    run_dir = make_run(tmp_path, run_config(tiny_model, data, maxSteps=6, every=3))
    status = finish(start(run_dir), run_dir)
    assert status["state"] == "done" and status["step"] == 6
    lines = metrics(run_dir)
    evals = [m for m in lines if m["type"] == "eval"]
    assert [m["step"] for m in evals] == [0, 3, 6]
    assert {"val_loss", "accuracy", "auroc"} <= set(evals[-1])
    assert all(np.isfinite(m["loss"]) for m in lines if m["type"] == "train")
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["step-0000003", "step-0000006"]
    assert (run_dir / "model" / "adapter" / "adapter_config.json").exists()
    assert json.loads((run_dir / "model" / "merged" / "anvil.json").read_text())["train_as"] == "decision"


def test_stop_saves_a_checkpoint_and_resume_finishes(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(400), decision_rows(8, seed=1))
    run_dir = make_run(tmp_path, run_config(tiny_model, data, maxSteps=400, every=1000))
    proc = start(run_dir)
    deadline = time.time() + 180
    while time.time() < deadline:
        status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
        if status.get("state") == "running" and status.get("step", 0) >= 3:
            break
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    stopped = finish(proc, run_dir)
    assert stopped["state"] == "stopped" and 3 <= stopped["step"] < 400, stopped
    assert (run_dir / "checkpoints" / f"step-{stopped['step']:07d}" / "state.json").exists()

    done = finish(start(run_dir, resume=True), run_dir)
    assert done["state"] == "done" and done["step"] == 400
    assert any(m["type"] == "resume" and m["step"] == stopped["step"] for m in metrics(run_dir))


def test_a_chat_run_learns_only_the_assistant(tmp_path, tiny_model):
    chats = pd.DataFrame({"messages": [json.dumps([{"role": "user", "content": f"hello {w}"},
                                                   {"role": "assistant", "content": f"thanks {w}"}]) for w in WORDS]})
    data = prepared(tmp_path, chats, chats.head(4))
    run_dir = make_run(tmp_path, run_config(tiny_model, data, train_as="lm", fmt="chat", maxSteps=3, every=3))
    status = finish(start(run_dir), run_dir)
    assert status["state"] == "done"
    assert "perplexity" in [m for m in metrics(run_dir) if m["type"] == "eval"][-1]


# ---------------------------------------------------------------- the pieces

def test_options_and_targets():
    noul = {"type": "noul", "criteria": {"true": "rude", "false": "polite"}}
    assert tasks.options(noul) == [(True, "yes: rude"), (False, "no: polite")]
    assert tasks.target(noul, False, tasks.options(noul)) == [0.0, 1.0]
    choice = {"type": "choice", "criteria": {"spam": "ads", "fine": None}}
    assert tasks.options(choice) == [("spam", "spam: ads"), ("fine", "fine")]
    score = {"type": "score", "criteria": ["mild", "harsh", "severe"]}
    assert tasks.target(score, 2, tasks.options(score)) == [0.0, 0.0, 1.0]
    soft = {"label": True, "probabilities": {"true": 0.75, "false": 0.25}}
    assert tasks.target(noul, soft, tasks.options(noul)) == [0.75, 0.25]


def test_shuffled_options_keep_targets_in_step(tiny_model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model)
    task = tasks.DecisionLetters(tok, 128, seed=3)
    df = decision_rows(30)
    items, parsed = task.items(df)
    a = task.collate(items, parsed, shuffle=True)
    b = task.collate(items, parsed, shuffle=True)
    assert (a["input_ids"] == b["input_ids"]).all() and (a["targets"] == b["targets"]).all()  # repeatable
    plain = task.collate(items, parsed, shuffle=False)
    assert (a["targets"].sum(-1) == 1).all() and not (a["input_ids"] == plain["input_ids"]).all()


def test_chat_labels_cover_only_assistant_turns(tiny_model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model)
    task = tasks.LanguageModel(tok, "chat", 256)
    messages = [{"role": "user", "content": "hello world"}, {"role": "assistant", "content": "thanks"},
                {"role": "user", "content": "please help"}, {"role": "assistant", "content": "fine"}]
    ids, labels = task._chat(messages)
    learned = tok.decode([t for t, label in zip(ids, labels) if label != -100])
    assert learned.replace(" ", "") == "thanks<|end|>fine<|end|>"


def test_auroc_matches_hand_counted_cases():
    assert tasks.auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([False, False, True, True])) == 0.75
    assert tasks.auroc(np.array([0.5, 0.5]), np.array([True, False])) == 0.5
    assert tasks.auroc(np.array([0.2, 0.3]), np.array([True, True])) is None


# ---------------------------------------------------------------- starting runs

FORM = {"name": "My run", "trainAs": "decision", "decisionMethod": "letters", "lmFormat": "chat",
        "trainingType": "lora", "model": {"source": "folder", "folder": "/m", "repo": None}, "gpus": None,
        "hyper": {"learningRate": "2e-4", "epochs": 1, "batchSize": 8, "gradAccum": 1, "maxLength": 512,
                  "maxSteps": 0},
        "checkpoints": {"every": 500}}


@pytest.fixture
def run_settings(tmp_path, monkeypatch):
    s = config.Settings(password="x", data=tmp_path / "data", hf_token=None)
    monkeypatch.setattr(config, "settings", s)
    monkeypatch.setattr(runs, "_model", lambda form, token: {"source": "folder", "id": "/m", "kind": "model",
                                                             "base_model": None, "custom_code": False})
    monkeypatch.setattr(runs.gpus, "list_gpus", lambda: {"gpus": [
        {"index": 0, "display": True, "busy": False}, {"index": 1, "display": False, "busy": False}]})
    spawned = []
    monkeypatch.setattr(runs, "_spawn", lambda run_id, path, gpus, token, resume, judge_gpu=None:
                        spawned.append((run_id, gpus, judge_gpu)))

    class Done:
        state, fingerprint = "done", "abc"
        report = {"format": "decision", "format_label": "Decision", "rows": {"train": 10}}

    monkeypatch.setattr(runs.datasets, "start", lambda request, token, run: Done())
    return spawned


def test_every_picked_card_trains_led_by_the_one_without_a_screen(run_settings):
    run_id = runs.create(FORM, {"train": {}}, None)  # no pick: every free card
    assert run_id.startswith("my-run-") and run_settings == [(run_id, [1, 0], None)]
    cfg = json.loads((config.settings.data / "runs" / run_id / "config.json").read_text())
    assert cfg["gpus"] == [1, 0] and cfg["gpu"] == 1 and cfg["judge_gpu"] is None
    assert cfg["data"]["prepared"].endswith("/abc")
    assert runs.summary(run_id)["gpus"] == [1, 0]


def test_a_free_card_you_didnt_pick_judges(run_settings):
    run_id = runs.create({**FORM, "gpus": [1]}, {"train": {}}, None)
    assert run_settings == [(run_id, [1], 0)]


def test_a_busy_card_neither_trains_unpicked_nor_judges(run_settings, monkeypatch):
    monkeypatch.setattr(runs.gpus, "list_gpus", lambda: {"gpus": [
        {"index": 0, "display": True, "busy": True}, {"index": 1, "display": False, "busy": False}]})
    run_id = runs.create(FORM, {"train": {}}, None)
    assert run_settings == [(run_id, [1], None)]


def test_cards_in_use_by_a_run_are_busy_until_it_ends(run_settings):
    run_id = runs.create(FORM, {"train": {}}, None)
    (config.settings.data / "runs" / run_id / "status.json").write_text(json.dumps(
        {"state": "running", "pid": os.getpid()}))  # this test process stands in for the trainer
    assert runs.busy_gpus() == {0: run_id, 1: run_id}
    with pytest.raises(ValueError, match="None of the picked GPUs is free"):
        runs.create(FORM, {"train": {}}, None)


def test_several_cards_start_through_torchrun():
    one = runs.command("anvil.train", Path("/data/runs/r"), 1, ["--resume"])
    assert one[1:] == ["-m", "anvil.train", "/data/runs/r", "--resume"]
    two = runs.command("anvil.train", Path("/data/runs/r"), 2, [])
    assert two[1:] == ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", "-m", "anvil.train",
                       "/data/runs/r"]


def test_a_judge_still_checking_after_stop_keeps_its_card_and_serves_the_resume(run_settings, monkeypatch):
    run_id = runs.create({**FORM, "gpus": [1]}, {"train": {}}, None)
    path = config.settings.data / "runs" / run_id
    me = os.getpid()  # stands in for live processes
    (path / "status.json").write_text(json.dumps({"state": "stopped", "step": 40}))
    (path / "judge" / "status.json").parent.mkdir(exist_ok=True)
    (path / "judge" / "status.json").write_text(json.dumps({"state": "checking", "gpu": 0, "pid": me,
                                                            "updated": time.time()}))
    (path / "prepared").mkdir()
    (path / "prepared" / "train.parquet").write_text("")
    cfg = json.loads((path / "config.json").read_text())
    cfg["data"]["prepared"] = str(path / "prepared")
    (path / "config.json").write_text(json.dumps(cfg))
    assert runs.busy_gpus() == {0: run_id}  # the trainer stopped, but its judge is still at work on GPU 0
    runs.resume(run_id, None)
    assert run_settings[-1] == (run_id, [1], None)  # no second judge: the one still running carries on
    assert json.loads((path / "config.json").read_text())["judge_gpu"] == 0


@pytest.mark.parametrize("change, reason", [
    ({"decisionMethod": "clef"}, "step 6"),
    ({"trainingType": "qlora"}, "QLoRA"),
    ({"trainAs": "lm"}, "needs chat"),
    ({"hyper": {**FORM["hyper"], "learningRate": "fast"}}, "isn't a number"),
])
def test_unsupported_runs_say_why(run_settings, change, reason):
    with pytest.raises(ValueError, match=reason):
        runs.create({**FORM, **change}, {"train": {}}, None)


def test_unprepared_data_is_refused(run_settings, monkeypatch):
    monkeypatch.setattr(runs.datasets, "start", lambda request, token, run: None)
    with pytest.raises(ValueError, match="Prepare the data first"):
        runs.create(FORM, {"train": {}}, None)


def test_a_dead_process_shows_as_failed(run_settings):
    run_id = runs.create(FORM, {"train": {}}, None)
    path = config.settings.data / "runs" / run_id
    status = json.loads((path / "status.json").read_text())
    (path / "status.json").write_text(json.dumps({**status, "state": "running", "pid": 999_999_999}))
    assert runs.status(run_id)["state"] == "failed"


# ---------------------------------------------------------------- the judge: picking, early stopping, a judge process

def test_an_unimprovable_run_stops_early_and_keeps_its_best(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(80), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, maxSteps=60, every=2, learningRate="1e-30", scheduler="constant")
    cfg["checkpoints"].update(patience=1, keepBest=1, pickBy="auroc")
    run_dir = make_run(tmp_path, cfg)
    status = finish(start(run_dir), run_dir)
    assert status["state"] == "done" and status["step"] == 4, status  # check at 2 is best, at 4 no better: stop
    assert "didn't improve for 1 checks" in status["early_stop"]
    assert status["best_step"] == 2 and status["final_test"]["step"] == 2
    board = json.loads((run_dir / "leaderboard.json").read_text())
    assert [e["step"] for e in board["entries"]] == [0, 2, 4] and board["best_step"] == 2
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["step-0000002", "step-0000004"]
    assert json.loads((run_dir / "model" / "merged" / "anvil.json").read_text())["picked_step"] == 2


def test_a_judge_process_checks_every_checkpoint(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, maxSteps=6, every=2)
    cfg["judge_gpu"] = 0
    run_dir = make_run(tmp_path, cfg)
    (run_dir / "judge").mkdir()
    env = {**os.environ, "ANVIL_DEVICE": "cpu", "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(SERVER)}
    judge = subprocess.Popen([sys.executable, "-m", "anvil.judge", str(run_dir)], env=env,
                             stdout=(run_dir / "judge" / "log.txt").open("ab"), stderr=subprocess.STDOUT)
    status = finish(start(run_dir), run_dir)
    assert judge.wait(timeout=240) == 0, (run_dir / "judge" / "log.txt").read_text()[-3000:]
    assert status["state"] == "done"
    by_judge = sorted(m["step"] for m in metrics(run_dir) if m["type"] == "eval" and m.get("by") == "judge")
    assert by_judge == [0, 2, 4, 6]
    assert not [m for m in metrics(run_dir) if m["type"] == "eval" and m.get("by") == "trainer"]
    assert json.loads((run_dir / "judge" / "status.json").read_text())["state"] == "done"
    assert status["final_test"]["step"] == json.loads((run_dir / "leaderboard.json").read_text())["best_step"]


def test_picking_rules():
    from anvil import picking

    board = {"entries": [], "best_step": None, "since_best": 0}
    for step, auroc in [(0, 0.9), (10, 0.7), (20, 0.8), (30, 0.75), (40, 0.79)]:
        board = picking.record(board, step, {"auroc": auroc, "val_loss": 1.0}, "auroc")
    assert board["best_step"] == 20 and board["since_best"] == 2  # step 0 scored highest but is never picked
    assert picking.should_stop(board, 2) and not picking.should_stop(board, 3) and not picking.should_stop(board, 0)
    assert [e["step"] for e in picking.ranked(board)] == [20, 40, 30, 10]
    assert picking.keep(board, 2, [10, 20, 30, 40, 50]) == {20, 40, 50}  # best two, plus the newest and unchecked
    assert picking.stop_reason(board, 2) == "Mean AUROC didn't improve for 2 checks in a row"
    board = picking.record(board, 50, {"auroc": 0.85}, "auroc")  # a later, better check cancels the stop
    assert board["best_step"] == 50 and picking.stop_reason(board, 2) is None
    loss_board = {"entries": [], "best_step": None, "since_best": 0}
    for step, loss in [(10, 0.5), (20, 0.4), (30, 0.45)]:
        loss_board = picking.record(loss_board, step, {"val_loss": loss}, "val_loss")
    assert loss_board["best_step"] == 20  # lower is better for losses
    no_auroc = picking.record({"entries": [], "best_step": None, "since_best": 0}, 10, {"val_loss": 0.3}, "auroc")
    assert no_auroc["entries"][0]["metric"] == "val_loss"  # falls back when the data has no yes/no questions


def test_catch_rate_and_calibration():
    scores = np.array([0.1, 0.2, 0.3, 0.95, 0.6, 0.7, 0.99])
    labels = np.array([False, False, False, False, True, True, True])
    assert tasks.catch_rate(scores, labels, 0.0) == pytest.approx(1 / 3)  # only 0.99 beats the highest "no" (0.95)
    assert tasks.catch_rate(scores, labels, 0.25) == 1.0  # one "no" of four may be flagged: the cut-off drops to 0.3
    perfect = tasks.calibration_error(np.array([0.9] * 10), np.array([True] * 9 + [False]))
    assert perfect == pytest.approx(0.0, abs=1e-9)
    overconfident = tasks.calibration_error(np.array([0.9] * 10), np.array([True] * 5 + [False] * 5))
    assert overconfident == pytest.approx(0.4)


def test_a_run_whose_judge_never_shows_up_checks_itself(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, maxSteps=6, every=2)
    cfg["judge_gpu"] = 0  # but no judge process: the trainer notices at the first checkpoint and takes over
    run_dir = make_run(tmp_path, cfg)
    status = finish(start(run_dir), run_dir)
    assert status["state"] == "done"
    assert sorted(m["step"] for m in metrics(run_dir) if m["type"] == "eval") == [2, 4, 6]
    assert "the judge stopped, so checks run here" in (run_dir / "log.txt").read_text()
    assert status["final_test"]["step"] == status["best_step"]


def test_the_judge_stops_training_early(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(80), decision_rows(12, seed=1))
    # A learning rate too small to change anything: the first trained check is the best, the next one is no better.
    cfg = run_config(tiny_model, data, maxSteps=20000, every=50, learningRate="1e-30", scheduler="constant")
    cfg["checkpoints"].update(patience=1, keepBest=1)
    cfg["judge_gpu"] = 0
    run_dir = make_run(tmp_path, cfg)
    (run_dir / "judge").mkdir()
    env = {**os.environ, "ANVIL_DEVICE": "cpu", "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(SERVER)}
    judge = subprocess.Popen([sys.executable, "-m", "anvil.judge", str(run_dir)], env=env,
                             stdout=(run_dir / "judge" / "log.txt").open("ab"), stderr=subprocess.STDOUT)
    status = finish(start(run_dir), run_dir, timeout=400)
    assert judge.wait(timeout=240) == 0, (run_dir / "judge" / "log.txt").read_text()[-3000:]
    assert status["state"] == "done" and status["step"] < 20000, status
    assert status["early_stop"] == "Mean AUROC didn't improve for 1 checks in a row" and status["best_step"] == 50
    judged = {m["step"] for m in metrics(run_dir) if m["type"] == "eval"}
    saved = {int(p.name[5:]) for p in (run_dir / "checkpoints").iterdir()}
    assert saved <= judged and 50 in saved  # everything left was checked; the best is still there
    assert len(saved) <= 2  # the best, plus the newest for Resume


def test_the_trainer_and_the_judge_never_overwrite_each_others_checks(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from anvil import picking

    def add(step):
        with picking.updating(tmp_path) as board:
            time.sleep(0.01)  # widen the window a lost update would need
            picking.record(board, step, {"val_loss": 1.0 / step}, "val_loss")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(add, range(1, 41)))
    board = picking.load(tmp_path)
    assert [e["step"] for e in board["entries"]] == list(range(1, 41))
    assert board["best_step"] == 40


# ---------------------------------------------------------------- several cards at once (two CPU processes)

def test_two_cards_learn_exactly_what_one_card_learns_from_the_whole_batch(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(60), decision_rows(12, seed=1))
    common = dict(maxSteps=6, every=3, gradCheckpointing=True)
    one = make_run(tmp_path, run_config(tiny_model, data, batchSize=4, **common), "one")
    two = make_run(tmp_path, run_config(tiny_model, data, batchSize=2, **common), "two")  # 2 per card × 2 cards
    finish(start(one), one)
    status = finish(start(two, cards=2), two)
    assert status["state"] == "done" and status["world"] == 2 and status["items_per_step"] == 4

    def by_step(run_dir, kind, key):
        return {m["step"]: m[key] for m in metrics(run_dir) if m["type"] == kind}

    a, b = by_step(one, "train", "loss"), by_step(two, "train", "loss")
    assert sorted(a) == sorted(b) == [1, 2, 3, 4, 5, 6]
    assert np.allclose([a[k] for k in sorted(a)], [b[k] for k in sorted(b)], rtol=1e-3, atol=1e-5), (a, b)
    va, vb = by_step(one, "eval", "val_loss"), by_step(two, "eval", "val_loss")
    assert sorted(va) == sorted(vb) == [0, 3, 6]  # checks are shared across the cards and add up to the same
    assert np.allclose([va[k] for k in sorted(va)], [vb[k] for k in sorted(vb)], rtol=1e-3, atol=1e-5), (va, vb)
    assert by_step(two, "eval", "rows")[6] == by_step(one, "eval", "rows")[6]
    assert (two / "model" / "adapter" / "adapter_config.json").exists()
    assert "Traceback" not in (two / "log.txt").read_text()


def test_two_cards_stop_together_and_resume(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(400), decision_rows(8, seed=1))
    run_dir = make_run(tmp_path, run_config(tiny_model, data, batchSize=2, maxSteps=200, every=1000))
    proc = start(run_dir, cards=2)
    deadline = time.time() + 180
    while time.time() < deadline:
        status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
        if status.get("state") == "running" and status.get("step", 0) >= 3:
            break
        time.sleep(0.05)
    os.kill(status["pid"], signal.SIGTERM)  # what Stop does: the signal reaches process 0 only
    stopped = finish(proc, run_dir)
    assert stopped["state"] == "stopped" and 3 <= stopped["step"] < 200, stopped
    state = json.loads((run_dir / "checkpoints" / f"step-{stopped['step']:07d}" / "state.json").read_text())
    assert state["world"] == 2
    done = finish(start(run_dir, resume=True, cards=2), run_dir)
    assert done["state"] == "done" and done["step"] == 200


def test_two_cards_stop_early_together(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(80), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, batchSize=2, maxSteps=60, every=2, learningRate="1e-30", scheduler="constant")
    cfg["checkpoints"].update(patience=1, keepBest=1)
    run_dir = make_run(tmp_path, cfg)
    status = finish(start(run_dir, cards=2), run_dir)
    assert status["state"] == "done" and status["step"] == 4 and status["best_step"] == 2, status
    assert status["early_stop"] == "Mean AUROC didn't improve for 1 checks in a row"


def test_two_cards_with_a_judge(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, batchSize=2, maxSteps=6, every=2)
    cfg["judge_gpu"] = 2
    run_dir = make_run(tmp_path, cfg)
    (run_dir / "judge").mkdir()
    env = {**os.environ, "ANVIL_DEVICE": "cpu", "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(SERVER)}
    judge = subprocess.Popen([sys.executable, "-m", "anvil.judge", str(run_dir)], env=env,
                             stdout=(run_dir / "judge" / "log.txt").open("ab"), stderr=subprocess.STDOUT)
    status = finish(start(run_dir, cards=2), run_dir)
    assert judge.wait(timeout=240) == 0, (run_dir / "judge" / "log.txt").read_text()[-3000:]
    assert status["state"] == "done"
    checked = {m["step"] for m in metrics(run_dir) if m["type"] == "eval"}
    assert checked == {0, 2, 4, 6}
    assert status["final_test"]["step"] == status["best_step"]
