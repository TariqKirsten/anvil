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


def start(run_dir: Path, resume=False) -> subprocess.Popen:
    env = {**os.environ, "ANVIL_DEVICE": "cpu", "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(SERVER)}
    log = (run_dir / "log.txt").open("ab")
    return subprocess.Popen([sys.executable, "-m", "anvil.train", str(run_dir)] + (["--resume"] if resume else []),
                            env=env, stdout=log, stderr=subprocess.STDOUT)


def make_run(tmp: Path, cfg: dict) -> Path:
    run_dir = tmp / "run"
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
    monkeypatch.setattr(runs, "_spawn", lambda run_id, path, gpu, token, resume: spawned.append((run_id, gpu)))

    class Done:
        state, fingerprint = "done", "abc"
        report = {"format": "decision", "format_label": "Decision", "rows": {"train": 10}}

    monkeypatch.setattr(runs.datasets, "start", lambda request, token, run: Done())
    return spawned


def test_a_run_starts_on_the_card_without_a_screen(run_settings):
    run_id = runs.create(FORM, {"train": {}}, None)
    assert run_id.startswith("my-run-") and run_settings == [(run_id, 1)]
    cfg = json.loads((config.settings.data / "runs" / run_id / "config.json").read_text())
    assert cfg["gpu"] == 1 and cfg["data"]["prepared"].endswith("/abc")


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
