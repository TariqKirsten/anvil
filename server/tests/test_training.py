"""Training end to end on the CPU, with a tiny random Llama and a tiny tokenizer made here, plus the pieces it uses."""

import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anvil import config, runs, supervisor, tasks, train

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
        num_key_value_heads=2, max_position_embeddings=2048, pad_token_id=tok.pad_token_id,
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


def start(run_dir: Path, resume=False, cards=1, supervised=False) -> subprocess.Popen:
    """The trainer, as the server starts it; several "cards" are several CPU processes (gloo instead of RCCL).
    supervised: under anvil.supervisor, which starts it again if it crashes."""
    env = {**os.environ, "ANVIL_DEVICE": "cpu", "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(SERVER), "OMP_NUM_THREADS": "2"}
    log = (run_dir / "log.txt").open("ab")
    cmd = runs.command("anvil.train", run_dir, cards, ["--resume"] if resume else [])
    if supervised:
        cmd = [sys.executable, "-m", "anvil.supervisor", str(run_dir), "--", *cmd]
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


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


def served(tmp: Path, monkeypatch, cfg: dict, run_id: str) -> Path:
    """A run in the server's runs folder, which runs._spawn starts as the server does (under the supervisor, which is
    what Stop and Resume talk to), only on the CPU."""
    monkeypatch.setattr(config, "settings", config.Settings(password="x", data=tmp / "data", hf_token=None))
    monkeypatch.setattr(runs, "_processes", {})

    def start_on_the_cpu(module, key, path, cards, token, log_path, args):
        runs._processes[key] = start(path, resume="--resume" in args, cards=len(cards), supervised=True)
        return runs._processes[key].pid

    monkeypatch.setattr(runs, "_start", start_on_the_cpu)
    (tmp / "data" / "runs").mkdir(parents=True)
    return make_run(tmp / "data" / "runs", cfg, run_id)


def wait_until(run_dir: Path, check, timeout=180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
        if check(status):
            return status
        time.sleep(0.05)
    raise AssertionError(f"timed out: {status}")


def gone(pid: int) -> bool:
    found = supervisor._stat(pid)
    return found is None or found[1] == "Z"


# ---------------------------------------------------------------- end to end

def test_a_decision_run_trains_checkpoints_and_saves(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    run_dir = make_run(tmp_path, run_config(tiny_model, data, maxSteps=6, every=3))
    status = finish(start(run_dir), run_dir)
    assert status["state"] == "done" and status["step"] == 6
    lines = metrics(run_dir)
    evals = [m for m in lines if m["type"] == "eval"]
    assert [m["step"] for m in evals] == [0, 3, 6]
    assert {"val_loss", "accuracy", "auroc", "macro_f1", "overall"} <= set(evals[-1])
    assert {name: q["type"] for name, q in evals[-1]["questions"].items()} == {"rude": "noul", "kind": "choice"}
    assert all(np.isfinite(q["loss"]) for q in evals[-1]["questions"].values())
    # half of validation's 12 rude and 4 kind questions, re-scored at every check once training has learned from them
    plan = json.loads((run_dir / "trained_rows.json").read_text())
    assert len(plan["items"]) == 6 + 2 and 1 <= plan["from"] <= 6
    assert [m["step"] for m in evals if "trained" in m] == [s for s in (0, 3, 6) if s >= plan["from"]]
    assert evals[-1]["trained"]["rows"] == 8 and set(evals[-1]["trained"]["questions"]) == {"rude", "kind"}
    assert set(status["final_test"]["trained"]["questions"]) == {"rude", "kind"}
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


def test_stop_ends_a_trainer_that_cant_finish_its_step_and_resume_finishes(tmp_path, tiny_model, monkeypatch):
    data = prepared(tmp_path, decision_rows(400), decision_rows(8, seed=1))
    run_dir = served(tmp_path, monkeypatch, run_config(tiny_model, data, maxSteps=400, every=1000), "wedged")
    runs._spawn("wedged", run_dir, [0], None, resume=False)
    wait_until(run_dir, lambda st: st.get("state") == "running" and st.get("step", 0) >= 3)
    runs.stop("wedged")  # through the supervisor: the trainer saves after this step and stops by itself
    stopped = finish(runs._processes["wedged"], run_dir)
    assert stopped["state"] == "stopped" and not stopped.get("ended"), stopped
    saved = stopped["step"]
    assert (run_dir / "checkpoints" / f"step-{saved:07d}" / "state.json").exists()

    runs.resume("wedged", None)
    st = wait_until(run_dir, lambda st: st.get("state") == "running" and st.get("step", 0) >= saved + 6)
    os.kill(st["pid"], signal.SIGSTOP)  # wedged, as in a GPU call: it can't act on anything but SIGKILL now
    monkeypatch.setattr(supervisor, "STOP_GRACE_S", 3)
    runs.stop("wedged")
    stopping = runs.status("wedged")["stopping"]
    assert not stopping["ending"] and stopping["end_by"] == pytest.approx(stopping["requested"] + 3)
    ended = finish(runs._processes["wedged"], run_dir)  # once the wait has run out, the supervisor ends it
    assert ended["state"] == "stopped" and "crashes" not in ended, ended
    assert ended["ended"]["from"] == saved and ended["ended"]["step"] > saved
    assert ended["ended"]["at"] >= stopping["requested"] + 3 and gone(st["pid"])

    runs.resume("wedged", None)  # from the newest checkpoint: the steps since are trained again
    done = finish(runs._processes["wedged"], run_dir)
    assert done["state"] == "done" and done["step"] == 400, done
    assert [m["step"] for m in metrics(run_dir) if m["type"] == "resume"] == [saved, saved]


def test_a_trainer_that_crashes_starts_again_from_its_newest_checkpoint(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(400), decision_rows(8, seed=1))
    cfg = run_config(tiny_model, data, maxSteps=400, every=5)
    cfg["checkpoints"].update(patience=0)
    run_dir = make_run(tmp_path, cfg)
    proc = start(run_dir, supervised=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
        if status.get("state") == "running" and status.get("step", 0) >= 12:
            break
        time.sleep(0.05)
    os.kill(status["pid"], signal.SIGKILL)  # as a crash in native code would: no chance to say how it ended
    done = finish(proc, run_dir)
    assert done["state"] == "done" and done["step"] == 400 and done["crashes"] == 1, done
    restart = done["last_crash"]["from"]
    assert restart >= 10 and restart % 5 == 0
    assert any(m["type"] == "resume" and m["step"] == restart for m in metrics(run_dir))
    assert "restarting from the step-" in (run_dir / "log.txt").read_text()


def poison(monkeypatch, change) -> None:
    """The task's loss for each training batch becomes change(call, loss, model); call counts the batches from 1."""
    from anvil import train

    make, calls = train.make_task, [0]

    def make_task(*args):
        task = make(*args)
        loss = task.loss_sum

        def changed(model, batch):
            calls[0] += 1
            return change(calls[0], loss(model, batch), model)

        task.loss_sum = changed
        return task

    monkeypatch.setattr(train, "make_task", make_task)


def test_a_step_whose_gradients_arent_numbers_is_skipped_and_the_weights_stay_sound(tmp_path, tiny_model, monkeypatch,
                                                                                     capsys):
    import torch

    from anvil import parallel, train

    def nan_gradients(call, loss, model):
        if call != 3:
            return loss
        p = next(q for q in model.parameters() if q.requires_grad)
        return loss + 0 * torch.sqrt(p - p.detach()).sum()  # adds 0 to the loss, but its gradient is 0/0: NaN

    poison(monkeypatch, nan_gradients)
    monkeypatch.setenv("ANVIL_DEVICE", "cpu")
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, maxSteps=6, every=3)
    run_dir = make_run(tmp_path, cfg)
    train.train(cfg, run_dir, train.Reporter(run_dir), False, parallel.Group())
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "done" and status["step"] == 6 and status["skipped"] == 1, status
    assert all(np.isfinite(m["val_loss"]) for m in metrics(run_dir) if m["type"] == "eval")
    assert "step 3: skipped batch 2 of epoch 0, its gradients weren't numbers (model nan)" in capsys.readouterr().out


def test_a_run_that_cant_get_past_bad_batches_fails_instead_of_skipping_forever(tmp_path, tiny_model, monkeypatch):
    from anvil import parallel, train

    poison(monkeypatch, lambda call, loss, model: loss if call <= 2 else loss * float("nan"))
    monkeypatch.setenv("ANVIL_DEVICE", "cpu")
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, maxSteps=6, every=3)
    run_dir = make_run(tmp_path, cfg)
    with pytest.raises(ValueError, match=f"{train.MAX_SKIPS} batches in a row"):
        train.train(cfg, run_dir, train.Reporter(run_dir), False, parallel.Group())
    status = json.loads((run_dir / "status.json").read_text())
    assert status["step"] == 2 and status["skipped"] == train.MAX_SKIPS, status


def test_a_chat_run_learns_only_the_assistant(tmp_path, tiny_model):
    chats = pd.DataFrame({"messages": [json.dumps([{"role": "user", "content": f"hello {w}"},
                                                   {"role": "assistant", "content": f"thanks {w}"}]) for w in WORDS]})
    data = prepared(tmp_path, chats, chats.head(4))
    run_dir = make_run(tmp_path, run_config(tiny_model, data, train_as="lm", fmt="chat", maxSteps=3, every=3))
    status = finish(start(run_dir), run_dir)
    assert status["state"] == "done"
    evals = [m for m in metrics(run_dir) if m["type"] == "eval"]
    assert {"perplexity", "token_accuracy", "samples"} <= set(evals[-1]) and 0 <= evals[-1]["token_accuracy"] <= 1
    # Sample replies: the same three held-out prompts at every check, answered by the model as it is then.
    assert [s["prompt"] for s in evals[-1]["samples"]] == [s["prompt"] for s in evals[0]["samples"]]
    assert len(evals[-1]["samples"]) == 3 and evals[-1]["samples"][0]["prompt"] == "user: hello the"
    assert set(evals[-1]["samples"][0]) == {"prompt", "expected", "reply", "ended", "tokens"}
    assert evals[-1]["samples"][0]["expected"] == "thanks the" and 0 < evals[-1]["samples"][0]["tokens"] <= 96
    assert "samples" not in status["last_eval"] and len(status["final_test"]["samples"]) == 3


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
    later = task.collate(items, parsed, shuffle=True, salt=1)  # another epoch: other letters, targets still in step
    assert not (later["input_ids"] == a["input_ids"]).all() and (later["targets"].sum(-1) == 1).all()
    assert a["count"] == len(items) and a["padded"] == a["input_ids"].numel() >= a["tokens"] and a["truncated"] == 0
    wordy = {"state": ["hello " * 100], "questions": [{"rude": {"type": "noul"}}], "gold": [{"rude": True}]}
    cut = tasks.DecisionLetters(tok, 24, seed=3).collate([tasks.Item(0, "rude")], wordy, shuffle=False)
    assert cut["truncated"] == 1 and cut["input_ids"].shape[1] < 100  # the state cut to fit (16 tokens at least)


def test_chat_labels_cover_only_assistant_turns(tiny_model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model)
    task = tasks.LanguageModel(tok, "chat", 256)
    messages = [{"role": "user", "content": "hello world"}, {"role": "assistant", "content": "thanks"},
                {"role": "user", "content": "please help"}, {"role": "assistant", "content": "fine"}]
    ids, labels, cut = task._chat(messages)
    learned = tok.decode([t for t, label in zip(ids, labels) if label != -100])
    assert learned.replace(" ", "") == "thanks<|end|>fine<|end|>" and not cut
    # A reply that repeats the user's words is learned from the assistant's turn, not found in the user's.
    echo = [{"role": "user", "content": "thanks"}, {"role": "assistant", "content": "thanks"}]
    ids, labels, _ = task._chat(echo)
    text = tok.decode(ids)
    learned_at = [i for i, label in enumerate(labels) if label != -100]
    assert tok.decode([ids[i] for i in learned_at]).replace(" ", "") == "thanks<|end|>"
    assert tok.decode(ids[: learned_at[0]]).replace(" ", "") == "<|user|>thanks<|end|><|assistant|>", text
    # A conversation longer than max length is cut, and says so.
    long = [{"role": "user", "content": "hello " * 50}, {"role": "assistant", "content": "thanks " * 50}]
    ids, labels, cut = tasks.LanguageModel(tok, "chat", 40)._chat(long)
    assert len(ids) == 40 and cut


def test_text_and_prompts_start_with_the_models_bos_token(tiny_model):
    from tokenizers import processors
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model)
    plain = tasks.LanguageModel(tok, "text", 8)
    assert plain.bos == []  # the tiny tokenizer, like Qwen's, puts nothing at the start
    ids, labels, cut = plain.example(0, {"text": ["hello world " * 20]})
    assert len(ids) == 8 and ids[-1] == tok.eos_token_id and labels == ids and cut
    # A tokenizer that starts every text with a beginning-of-text token, like Llama's or Gemma's.
    bos = AutoTokenizer.from_pretrained(tiny_model)
    bos.add_special_tokens({"bos_token": "<|user|>"})
    bos.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<|user|> $A", special_tokens=[("<|user|>", bos.bos_token_id)])
    assert bos("x").input_ids[0] == bos.bos_token_id
    text = tasks.LanguageModel(bos, "text", 8)
    ids, labels, cut = text.example(0, {"text": ["hello world " * 20]})
    assert ids[0] == bos.bos_token_id and ids[-1] == bos.eos_token_id and len(ids) == 8 and cut
    pairs = tasks.LanguageModel(bos, "pairs", 64)
    ids, labels, cut = pairs.example(0, {"prompt": ["hello"], "completion": ["world"]})
    assert ids[0] == bos.bos_token_id and labels[0] == -100 and ids[-1] == bos.eos_token_id and not cut
    assert bos.decode([t for t in labels if t != -100]).replace(" ", "") == "world<|end|>"


def test_a_steps_loss_weighs_every_token_the_same_whatever_batch_it_is_in(tmp_path, tiny_model, monkeypatch):
    """Two batches a step, one of one-word replies and one of long ones: the step's loss (and gradient) is the mean
    over all their tokens, not the mean of the two batches' means, which would weigh a short reply's tokens more."""
    from anvil import parallel, train

    short = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "yes"}]
    long = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": " ".join(WORDS * 2)}]
    chats = pd.DataFrame({"messages": [json.dumps(short if i % 2 else long) for i in range(16)]})
    data = prepared(tmp_path, chats, chats.head(4))
    cfg = run_config(tiny_model, data, train_as="lm", fmt="chat", maxSteps=3, every=3, batchSize=2, gradAccum=2)
    run_dir = make_run(tmp_path, cfg)
    sums, make = [], train.make_task

    def make_task(*args):
        task = make(*args)
        loss_sum = task.loss_sum

        def recorded(model, batch):
            loss = loss_sum(model, batch)
            sums.append((float(loss.detach()), batch["count"], batch["tokens"], batch["padded"]))
            return loss

        task.loss_sum = recorded
        return task

    monkeypatch.setattr(train, "make_task", make_task)
    monkeypatch.setenv("ANVIL_DEVICE", "cpu")
    train.train(cfg, run_dir, train.Reporter(run_dir), False, parallel.Group())
    lines = [m for m in metrics(run_dir) if m["type"] == "train"]
    assert [m["step"] for m in lines] == [1, 2, 3]
    for step, entry in enumerate(lines):
        batches = sums[2 * step: 2 * step + 2]
        counts = [c for _, c, _, _ in batches]
        assert len(batches) == 2 and min(counts) * 4 < max(counts)  # one batch of short replies, one of long
        by_token = sum(s for s, *_ in batches) / sum(counts)
        by_batch = np.mean([s / c for s, c, _, _ in batches])
        assert entry["loss"] == pytest.approx(by_token, rel=1e-5) and abs(by_token - by_batch) > 1e-3
        assert entry["padding"] == pytest.approx(1 - sum(t for _, _, t, _ in batches) / sum(p for *_, p in batches))
        assert entry["tokens"] == sum(t for _, _, t, _ in sums[: 2 * step + 2])
        assert entry["grad_norm"] > 0 and entry["grad_norm_max"] >= entry["grad_norm"]
    status = json.loads((run_dir / "status.json").read_text())
    assert status["tokens_seen"] == lines[-1]["tokens"] and status["truncated"] == 0 and status["grad_norm"] > 0
    state = json.loads((run_dir / "checkpoints" / "step-0000003" / "state.json").read_text())
    assert state["tokens"] == status["tokens_seen"]


def test_a_full_fine_tune_trains_bf16_weights_with_stochastic_rounding(tmp_path, tiny_model):
    import torch
    from safetensors.torch import load_file

    chats = pd.DataFrame({"messages": [json.dumps([{"role": "user", "content": f"hello {w}"},
                                                   {"role": "assistant", "content": f"thanks {w}"}]) for w in WORDS]})
    data = prepared(tmp_path, chats, chats.head(4))
    cfg = run_config(tiny_model, data, train_as="lm", fmt="chat", maxSteps=6, every=3, learningRate="1e-4")
    cfg["training_type"] = "full"
    run_dir = make_run(tmp_path, cfg)
    status = finish(start(run_dir), run_dir)
    assert status["state"] == "done" and status["step"] == 6
    saved = load_file(str(run_dir / "checkpoints" / "step-0000006" / "model" / "model.safetensors"))
    before = load_file(str(tiny_model / "model.safetensors"))
    assert all(v.dtype == torch.bfloat16 for v in saved.values())  # the weights stayed bf16
    moved = sum((saved[k].float() != before[k].float()).sum().item() for k in saved) / sum(v.numel() for v in saved.values())
    assert moved > 0.3, moved  # at 1e-4 on bf16 weights, most updates would round away without the rounding
    assert "optimizer" in torch.load(run_dir / "checkpoints" / "step-0000006" / "optimizer.pt", weights_only=False)
    assert (run_dir / "model" / "merged" / "model.safetensors").exists()


def test_rows_cut_to_fit_max_length_are_counted(tmp_path, tiny_model, monkeypatch):
    from anvil import parallel, train

    from transformers import AutoTokenizer

    chats = pd.DataFrame({"messages": [json.dumps([{"role": "user", "content": "hello"},
                                                   {"role": "assistant", "content": " ".join(WORDS * 4 if i % 2 else ["yes"])}])
                                       for i in range(8)]})
    data = prepared(tmp_path, chats, chats.head(2))
    cfg = run_config(tiny_model, data, train_as="lm", fmt="chat", maxSteps=2, every=2, batchSize=4, maxLength=48)
    run_dir = make_run(tmp_path, cfg)
    monkeypatch.setenv("ANVIL_DEVICE", "cpu")
    train.train(cfg, run_dir, train.Reporter(run_dir), False, parallel.Group())
    task = tasks.LanguageModel(AutoTokenizer.from_pretrained(tiny_model), "chat", 48)
    items, parsed = task.items(chats)
    assert [task.example(i, parsed)[2] for i in items] == [False, True] * 4
    status = json.loads((run_dir / "status.json").read_text())
    assert status["truncated"] == 4  # the four long rows: all eight were trained on in two steps of four


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


def test_checks_can_stay_on_the_training_card_to_keep_the_spare_free(run_settings):
    run_id = runs.create({**FORM, "gpus": [1], "spareChecks": False}, {"train": {}}, None)
    assert run_settings == [(run_id, [1], None)]
    other = runs.create({**FORM, "gpus": [0], "spareChecks": False}, {"train": {}}, None)  # the other card is free
    assert run_settings[-1] == (other, [0], None)


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


def test_a_crashed_trainer_still_runs_while_its_supervisor_starts_it_again(run_settings):
    run_id = runs.create(FORM, {"train": {}}, None)
    path = config.settings.data / "runs" / run_id
    # Stands in for the supervisor: what the server checks is its command line, which names the module and the run.
    stand_in = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "anvil.supervisor", str(path)])
    try:
        (path / "status.json").write_text(json.dumps({"state": "running", "step": 40, "pid": 999_999_999,
                                                      "supervisor": stand_in.pid}))
        assert runs.status(run_id)["state"] == "running"  # the trainer is gone, but not the run
        assert run_id in runs.busy_gpus().values()  # its cards stay taken for the restart
        runs.stop(run_id)  # the trainer can't hear it: the supervisor does, and starts nothing new
        assert stand_in.wait(timeout=10) == -signal.SIGTERM
        assert runs.status(run_id)["state"] == "stopped"  # with neither left the run is over, stopped as asked
    finally:
        stand_in.kill()


def test_stop_reaches_process_0_and_stop_now_the_supervisor_as_runs_from_before_understand(run_settings):
    # The supervisor reads stop.json, but one started before it did would pass a SIGTERM of its own on to torchrun,
    # which kills the cards 30 s later, perhaps in the middle of a save: Stop goes to process 0, as it always has.
    run_id = runs.create(FORM, {"train": {}}, None)
    path = config.settings.data / "runs" / run_id
    trainer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    stand_in = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "anvil.supervisor", str(path)])
    try:
        (path / "status.json").write_text(json.dumps({"state": "running", "step": 40, "pid": trainer.pid,
                                                      "supervisor": stand_in.pid}))
        runs.stop(run_id)
        assert trainer.wait(timeout=10) == -signal.SIGTERM and stand_in.poll() is None
        asked = json.loads((path / "stop.json").read_text())
        assert asked["end_by"] == pytest.approx(asked["requested"] + supervisor.STOP_GRACE_S)
        runs.stop(run_id, now=True)  # an old supervisor ends a wedged run of several cards through torchrun
        assert stand_in.wait(timeout=10) == -signal.SIGTERM
        assert json.loads((path / "stop.json").read_text())["end_by"] <= time.time()
        st = runs.status(run_id)  # gone after Stop without saying so: stopped, not failed
        assert st["state"] == "stopped" and st["ended"]["step"] == 40 and st["ended"]["from"] is None, st
    finally:
        trainer.kill()
        stand_in.kill()


def test_deleting_a_run_removes_everything_it_saved(run_settings):
    run_id = runs.create(FORM, {"train": {}}, None)
    path = config.settings.data / "runs" / run_id
    (path / "status.json").write_text(json.dumps({"state": "done"}))
    (path / "checkpoints" / "step-0000010").mkdir(parents=True)
    (path / "checkpoints" / "step-0000010" / "adapter_model.safetensors").write_bytes(b"x" * 1000)
    assert runs.summary(run_id)["size"] >= 1000
    assert runs.delete(run_id) >= 1000 and not path.exists()
    with pytest.raises(FileNotFoundError):
        runs.delete(run_id)


def test_a_running_run_or_its_judge_blocks_deleting(run_settings):
    run_id = runs.create({**FORM, "gpus": [1]}, {"train": {}}, None)  # GPU 0 judges
    path = config.settings.data / "runs" / run_id
    (path / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid()}))
    with pytest.raises(ValueError, match="still going"):
        runs.delete(run_id)
    (path / "status.json").write_text(json.dumps({"state": "stopped"}))
    (path / "judge").mkdir(exist_ok=True)
    (path / "judge" / "status.json").write_text(json.dumps({"state": "checking", "gpu": 0, "pid": os.getpid(),
                                                            "updated": time.time()}))
    with pytest.raises(ValueError, match="judge is still checking"):
        runs.delete(run_id)
    (path / "judge" / "status.json").write_text(json.dumps({"state": "done", "gpu": 0}))
    runs.delete(run_id)
    assert not path.exists()


@pytest.mark.parametrize("change, reason", [
    ({"hyper": {**FORM["hyper"], "headLearningRate": "big"}}, "head learning rate isn't a number"),
    ({"trainingType": "qlora"}, "QLoRA"),
    ({"trainAs": "lm"}, "needs chat"),
    ({"hyper": {**FORM["hyper"], "learningRate": "fast"}}, "isn't a number"),
    ({"checkpoints": {**FORM["checkpoints"], "minGain": -1}}, "least gain that counts"),
    ({"checkpoints": {**FORM["checkpoints"], "minGain": "lots"}}, "least gain that counts"),
    ({"hyper": {**FORM["hyper"], "masterWeights": "fp64"}}, "bf16 or fp32"),
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


def test_early_stopping_counts_only_the_checks_after_the_warmup(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(80), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, maxSteps=60, every=2, learningRate="1e-30", scheduler="constant",
                     warmupPercent=10)
    cfg["checkpoints"].update(patience=1, keepBest=1, pickBy="auroc")
    run_dir = make_run(tmp_path, cfg)
    status = finish(start(run_dir), run_dir)
    # Warming up until step 6: the checks at 4 and 6 don't count, so the first one without a gain is at 8.
    assert status["warm_until"] == 6 and status["step"] == 8 and status["best_step"] == 2, status
    assert status["early_stop"] == "Mean AUROC didn't improve for 1 checks in a row"


def test_a_run_that_stopped_early_trains_on_with_more_patience_or_to_the_end(tmp_path, tiny_model, monkeypatch):
    monkeypatch.setattr(config, "settings", config.Settings(password="x", data=tmp_path / "data", hf_token=None))
    monkeypatch.setattr(runs.gpus, "list_gpus", lambda: {"gpus": [{"index": 0, "display": False, "busy": False}]})
    procs = []
    monkeypatch.setattr(runs, "_spawn", lambda run_id, path, gpus, token, resume, judge_gpu=None:
                        procs.append(start(path, resume=resume)))
    data = prepared(tmp_path, decision_rows(80), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, maxSteps=20, every=2, learningRate="1e-30", scheduler="constant")
    cfg["checkpoints"].update(patience=1, keepBest=1, pickBy="auroc")
    (tmp_path / "data" / "runs").mkdir(parents=True)
    run_dir = make_run(tmp_path / "data" / "runs", cfg, "early")
    assert finish(start(run_dir), run_dir)["step"] == 4  # nothing improves after the check at 2

    with pytest.raises(ValueError, match="how many more checks"):
        runs.resume("early", None)
    runs.resume("early", None, patience=2)  # two more checks without a gain: 6 and 8
    checkpoints = json.loads((run_dir / "config.json").read_text())["checkpoints"]
    assert checkpoints["patience"] == 2 and checkpoints["countFrom"] == 4  # counted from where it picks up
    status = finish(procs[-1], run_dir)
    assert status["state"] == "done" and status["step"] == 8, status
    assert status["early_stop"] == "Mean AUROC didn't improve for 2 checks in a row"
    assert any(m["type"] == "resume" and m["step"] == 4 for m in metrics(run_dir))

    runs.resume("early", None, patience=0)  # to the last step
    status = finish(procs[-1], run_dir)
    assert status["state"] == "done" and status["step"] == 20 and status["early_stop"] is None, status
    assert status["best_step"] == 2 and status["final_test"]["step"] == 2
    with pytest.raises(ValueError, match="has finished"):
        runs.resume("early", None, patience=2)


def test_a_finished_run_trains_on_with_more_epochs_and_its_learning_rate_climbs_back(tmp_path, tiny_model,
                                                                                    monkeypatch):
    monkeypatch.setattr(config, "settings", config.Settings(password="x", data=tmp_path / "data", hf_token=None))
    monkeypatch.setattr(runs.gpus, "list_gpus", lambda: {"gpus": [{"index": 0, "display": False, "busy": False}]})
    procs = []
    monkeypatch.setattr(runs, "_spawn", lambda run_id, path, gpus, token, resume, judge_gpu=None:
                        procs.append(start(path, resume=resume)))
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))  # 54 questions: 14 steps of 4
    cfg = run_config(tiny_model, data, every=7, warmupPercent=30)  # one epoch; warming up for 4 steps
    cfg["checkpoints"].update(patience=0)
    (tmp_path / "data" / "runs").mkdir(parents=True)
    run_dir = make_run(tmp_path / "data" / "runs", cfg, "more")
    first = finish(start(run_dir), run_dir)
    assert first["state"] == "done" and first["step"] == first["total"] == 14 and first["warm_until"] == 4, first

    with pytest.raises(ValueError, match="has finished. Give it more epochs"):
        runs.resume("more", None, patience=0)
    with pytest.raises(ValueError, match="trained 1 epoch already, to step 14: give it at least 2"):
        runs.resume("more", None, patience=0, epochs=1)
    with pytest.raises(ValueError, match="how many more checks"):
        runs.resume("more", None, epochs=2)
    runs.resume("more", None, patience=0, epochs=2)
    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["hyper"]["epochs"] == 2 and saved["first_total"] == 14
    assert runs.status("more")["total"] == 28  # the run page shows the new length before the trainer has loaded
    status = finish(procs[-1], run_dir)
    assert status["state"] == "done" and status["step"] == status["total"] == 28, status
    assert status["warm_until"] == 4  # the warmup stays as it was: only the end of the schedule moved
    assert {"type": "resume", "step": 14, "total": 28} in metrics(run_dir)
    lr = {m["step"]: m["lr"] for m in metrics(run_dir) if m["type"] == "train"}
    assert lr[14] < 1e-6 * lr[18]  # one epoch's cosine had decayed to nothing...
    assert lr[15] < lr[16] < lr[17] < lr[18]  # ...so it climbs back over 4 steps, the warmup's length...
    assert all(lr[s] > lr[s + 1] for s in range(18, 27))  # ...to the cosine over two epochs, and down it again
    assert "the learning rate climbs from" in (run_dir / "log.txt").read_text()


def test_resume_with_new_epochs_moves_only_the_end_of_the_run(run_settings):
    run_id = runs.create({**FORM, "hyper": {**FORM["hyper"], "maxSteps": 600}}, {"train": {}}, None)
    path = config.settings.data / "runs" / run_id
    (path / "prepared").mkdir()
    (path / "prepared" / "train.parquet").write_text("")
    cfg = json.loads((path / "config.json").read_text())
    cfg["data"]["prepared"] = str(path / "prepared")
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "status.json").write_text(json.dumps({"state": "stopped", "step": 520, "total": 600,
                                                  "steps_per_epoch": 400}))
    (path / "checkpoints" / "step-0000500").mkdir(parents=True)
    (path / "checkpoints" / "step-0000500" / "state.json").write_text("{}")
    for bad in (0, 101, True, 2.5):
        with pytest.raises(ValueError, match="whole number of epochs from 1 to 100"):
            runs.resume(run_id, None, epochs=bad)
    with pytest.raises(ValueError, match=r"trained 1\.25 epochs already, to step 500: give it at least 2"):
        runs.resume(run_id, None, epochs=1)
    runs.resume(run_id, None, epochs=3)
    saved = json.loads((path / "config.json").read_text())
    assert saved["hyper"]["epochs"] == 3 and saved["hyper"]["maxSteps"] == 0  # the epochs lift Max steps
    assert saved["first_total"] == 600 and saved["checkpoints"].get("countFrom") is None
    assert json.loads((path / "status.json").read_text())["total"] == 1200
    assert run_settings[-1][0] == run_id
    (path / "status.json").write_text(json.dumps({"state": "stopped", "step": 520, "total": 1200,
                                                  "steps_per_epoch": 400}))
    runs.resume(run_id, None, epochs=4)
    assert json.loads((path / "config.json").read_text())["first_total"] == 600  # still the length it had first


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
    start_step = json.loads((run_dir / "trained_rows.json").read_text())["from"]
    assert [m["step"] for m in metrics(run_dir) if m["type"] == "eval" and "trained" in m] == \
        [s for s in (0, 2, 4, 6) if s >= start_step]
    assert not [m for m in metrics(run_dir) if m["type"] == "eval" and m.get("by") == "trainer"]
    assert json.loads((run_dir / "judge" / "status.json").read_text())["state"] == "done"
    assert status["final_test"]["step"] == json.loads((run_dir / "leaderboard.json").read_text())["best_step"]


def test_the_trained_rows_are_the_first_training_learns_from_as_validation_mixes_them():
    class Task:  # an item is (row, kind)
        @staticmethod
        def kind(item, parsed):
            return item[1]

    items = [(i, "b" if i % 3 == 0 else "a") for i in range(60)]
    val = [(0, "a")] * 8 + [(0, "b")] * 2
    lengths = np.ones(len(items))
    picked, since = train.trained_rows(Task(), items, {}, lengths, val, {}, per_batch=4, accum=2, seed=1)
    batches = tasks.length_grouped_batches(lengths, 4, random.Random(1))  # the first epoch's batches, as training makes
    order = [i for batch in batches for i in batch]
    first = {kind: [i for i in order if items[i][1] == kind] for kind in "ab"}
    assert sorted(picked) == sorted(first["a"][:4] + first["b"][:1])  # half of validation's 8 a and 2 b
    last = max(next(n for n, batch in enumerate(batches) if i in batch) for i in picked)
    assert since == last // 2 + 1  # two batches a step: training has learned from all of them by this step


def test_picking_rules():
    from anvil import picking

    board = {"entries": [], "best_step": None}
    for step, auroc in [(0, 0.9), (10, 0.7), (20, 0.8), (30, 0.75), (40, 0.79)]:
        board = picking.record(board, step, {"auroc": auroc, "val_loss": 1.0}, "auroc")
    assert board["best_step"] == 20 and picking.checks_without_gain(board) == 2  # step 0 is never picked
    assert picking.should_stop(board, 2) and not picking.should_stop(board, 3) and not picking.should_stop(board, 0)
    assert picking.checks_without_gain(board, after=30) == 1 and not picking.should_stop(board, 2, after=30)
    assert [e["step"] for e in picking.ranked(board)] == [20, 40, 30, 10]
    assert picking.keep(board, 2, [10, 20, 30, 40, 50]) == {20, 40, 50}  # best two, plus the newest and unchecked
    assert picking.stop_reason(board, 2) == "Mean AUROC didn't improve for 2 checks in a row"
    board = picking.record(board, 50, {"auroc": 0.85}, "auroc")  # a later, better check cancels the stop
    assert board["best_step"] == 50 and picking.stop_reason(board, 2) is None
    loss_board = {"entries": [], "best_step": None}
    for step, loss in [(10, 0.5), (20, 0.4), (30, 0.45)]:
        loss_board = picking.record(loss_board, step, {"val_loss": loss}, "val_loss")
    assert loss_board["best_step"] == 20  # lower is better for losses
    no_auroc = picking.record({"entries": [], "best_step": None}, 10, {"val_loss": 0.3}, "auroc")
    assert no_auroc["entries"][0]["metric"] == "val_loss"  # falls back when no yes/no question could be scored
    mae_board = {"entries": [], "best_step": None}
    for step, mae in [(10, 0.6), (20, 0.3), (30, 0.4)]:
        mae_board = picking.record(mae_board, step, {"mae": mae, "val_loss": 1.0}, "mae")
    assert mae_board["best_step"] == 20  # fewer levels off is better
    assert picking.problem("overall", {"noul"}) is None and picking.problem("auroc", None) is None
    assert "needs choice questions" in picking.problem("macro_f1", {"noul", "score"})
    assert "Unknown" in picking.problem("vibes", {"noul"})
    # The least gain that counts: a rise within it is neither a new best nor a fresh start for the no-gain count.
    board = {"entries": [], "best_step": None}
    for step, auroc in [(10, 0.70), (20, 0.705), (30, 0.709), (40, 0.72)]:
        board = picking.record(board, step, {"auroc": auroc, "se": {"auroc": 0.02}}, "auroc", min_gain=0.01)
    assert board["best_step"] == 40 and picking.checks_without_gain(board) == 0
    assert board["entries"][0]["se"] == 0.02 and "se" not in picking.record(board, 50, {"auroc": 0.5}, "auroc")["entries"][-1]
    board = picking.record(board, 60, {"auroc": 0.725}, "auroc", min_gain=0.01)
    assert board["best_step"] == 40 and picking.checks_without_gain(board) == 2  # 0.725 is within 0.01 of 0.72
    assert picking.stop_reason(board, 2, min_gain=0.01) == "Mean AUROC didn't improve by at least 0.01 for 2 checks in a row"
    assert picking.min_gain({"checkpoints": {}}) == 0 and picking.min_gain({"checkpoints": {"minGain": 0.005}}) == 0.005


def test_scores_come_with_their_standard_errors():
    rng = np.random.default_rng(1)
    gold = rng.random(400) < 0.3
    yes = np.clip(gold * 0.4 + rng.random(400) * 0.6, 0, 1)
    out = {"loss": list(-np.log(np.where(gold, yes, 1 - yes))), "correct": list((yes >= 0.5) == gold),
           "confidence": list(np.maximum(yes, 1 - yes)), "questions": {}}
    tasks.note(out["questions"], "rude", "noul", yes=yes.tolist(), gold=gold.tolist())
    out["questions"][("rude", "noul")] = {"yes": yes.tolist(), "gold": gold.tolist()}
    picked = rng.choice(["a", "b", "c"], 60)
    out["questions"][("kind", "choice")] = {"picked": picked.tolist(), "gold": rng.choice(["a", "b", "c"], 60).tolist()}
    out["loss"] += [0.5] * 60
    out["correct"] += [True] * 60
    out["confidence"] += [0.9] * 60
    scores = tasks.decision_metrics(out, 460, 0.01)
    se = scores["se"]
    assert {"val_loss", "accuracy", "ece", "auroc", "catch_rate", "false_alarms", "macro_f1", "overall"} <= set(se)
    assert 0.01 < se["auroc"] < 0.06 and 0.02 < se["macro_f1"] < 0.12 and se["val_loss"] > 0
    assert se == tasks.standard_errors(out, 0.01)  # seeded: the trainer and the judge get the same numbers
    few = {"loss": [0.1], "correct": [True], "confidence": [0.9], "questions": {}}
    assert tasks.decision_metrics(few, 1, 0.01)["se"] == {}
    rows = {"rows": [[2.0, 4, 1], [1.0, 2, 1], [6.0, 4, 0]] * 10}
    lm = tasks.LanguageModel.metrics(rows, 30)
    assert lm["val_loss"] == pytest.approx(9 / 10) and lm["token_accuracy"] == pytest.approx(0.2)
    assert 0 < lm["se"]["val_loss"] < 0.5 and 0 < lm["se"]["token_accuracy"] < 0.2
    assert tasks.LanguageModel.combine([{"rows": [[1.0, 1, 1]]}, {"rows": [[2.0, 2, 0]]}]) == {"rows": [[1.0, 1, 1], [2.0, 2, 0]]}


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


def test_a_two_card_run_carries_on_on_one_card_and_learns_what_two_would(tmp_path, tiny_model):
    data = prepared(tmp_path, decision_rows(60), decision_rows(12, seed=1))
    two = make_run(tmp_path, run_config(tiny_model, data, batchSize=2, maxSteps=6, every=3, gradCheckpointing=True),
                   "two")
    finish(start(two, cards=2), two)
    one = tmp_path / "one"  # the same run back at its step-3 checkpoint, carrying on on one card (its other one faults)
    shutil.copytree(two, one)
    shutil.rmtree(one / "checkpoints" / "step-0000006")
    status = finish(start(one, resume=True), one)
    assert status["state"] == "done" and status["step"] == 6 and status["world"] == 1, status
    assert "each taking 2 of every batch's 2 shares in turn" in (one / "log.txt").read_text()
    state = json.loads((one / "checkpoints" / "step-0000006" / "state.json").read_text())
    assert state["world"] == 2  # still split two ways, so it can go back to two cards

    def losses(run_dir, after_resume):
        lines = metrics(run_dir)
        start_at = max((i for i, m in enumerate(lines) if m["type"] == "resume"), default=-1) if after_resume else -1
        return {m["step"]: m["loss"] for m in lines[start_at + 1:] if m["type"] == "train" and m["step"] > 3}

    a, b = losses(two, False), losses(one, True)
    assert sorted(a) == sorted(b) == [4, 5, 6]
    assert np.allclose([a[k] for k in (4, 5, 6)], [b[k] for k in (4, 5, 6)], rtol=1e-3, atol=1e-5), (a, b)


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
    os.kill(status["pid"], signal.SIGTERM)  # reaching process 0 only (from a shell, say), it stops every card
    stopped = finish(proc, run_dir)
    assert stopped["state"] == "stopped" and 3 <= stopped["step"] < 200, stopped
    state = json.loads((run_dir / "checkpoints" / f"step-{stopped['step']:07d}" / "state.json").read_text())
    assert state["world"] == 2
    done = finish(start(run_dir, resume=True, cards=2), run_dir)
    assert done["state"] == "done" and done["step"] == 200


def test_two_cards_one_wedged_stop_now_ends_them_all_and_resume_finishes(tmp_path, tiny_model, monkeypatch):
    data = prepared(tmp_path, decision_rows(400), decision_rows(8, seed=1))
    cfg = {**run_config(tiny_model, data, batchSize=2, maxSteps=200, every=1000), "gpus": [0, 1]}
    run_dir = served(tmp_path, monkeypatch, cfg, "wedged")
    runs._spawn("wedged", run_dir, [0, 1], None, resume=False)
    wait_until(run_dir, lambda st: st.get("state") == "running" and st.get("step", 0) >= 3)
    runs.stop("wedged")  # the supervisor passes it to each card's process, and they stop together
    stopped = finish(runs._processes["wedged"], run_dir)
    assert stopped["state"] == "stopped" and not stopped.get("ended"), stopped
    saved = stopped["step"]
    assert json.loads((run_dir / "checkpoints" / f"step-{saved:07d}" / "state.json").read_text())["world"] == 2

    runs.resume("wedged", None)
    st = wait_until(run_dir, lambda st: st.get("state") == "running" and st.get("step", 0) >= saved + 6)
    procs = supervisor._processes()
    torchrun = next(pid for pid, (parent, _, _) in procs.items() if parent == runs._processes["wedged"].pid)
    cards = [pid for pid, (parent, _, _) in procs.items() if parent == torchrun]
    assert st["pid"] in cards and len(cards) == 2
    # The other card wedges, as a hung GPU would: process 0 waits for it in the next collective, which it can't leave
    # to act on Stop (for 30 minutes, until the collective times out).
    os.kill(next(pid for pid in cards if pid != st["pid"]), signal.SIGSTOP)
    runs.stop("wedged")
    with pytest.raises(subprocess.TimeoutExpired):
        runs._processes["wedged"].wait(timeout=3)
    assert runs.status("wedged")["stopping"]["ending"] is False
    runs.stop("wedged", now=True)  # the second press
    ended = finish(runs._processes["wedged"], run_dir)
    assert ended["state"] == "stopped" and ended["ended"]["from"] == saved and "crashes" not in ended, ended
    assert all(gone(pid) for pid in [torchrun, *cards])

    runs.resume("wedged", None)
    done = finish(runs._processes["wedged"], run_dir)
    assert done["state"] == "done" and done["step"] == 200, done
    assert [m["step"] for m in metrics(run_dir) if m["type"] == "resume"] == [saved, saved]


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
