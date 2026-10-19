"""What the model learns from a row, and how its loss and validation scores are computed.

Three tasks:
- DecisionLetters: a decision model the letters way. Each (state, question) becomes a prompt that lists the options as
  A, B, C... and ends at "Answer:". The model's next-token scores for the option letters are its answer, so nothing is
  generated. Option order is shuffled per row (seeded) while training, so no letter is favoured. One question per pass.
- DecisionClef: a decision model the Clef way (Cloudflare's joint schema head, anvil/vendor/clef). A row's state and all
  its questions are one prompt, and a small new network, the head, scores every option. One pass per row.
- LanguageModel: ordinary next-token training on chat (only the assistant's turns are learned), prompt and completion
  (only the completion), or plain text (all of it).

Both read the transformer's last hidden states and multiply only the rows of the output layer they need, instead of
computing scores for the whole vocabulary at every position (Qwen3.5's 248k-word vocabulary makes that huge).
"""

from __future__ import annotations

import json
import math
import random
import zlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOSS_CHUNK = 2048  # positions scored at once by the language-model loss


def _value(value):
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def decoder_and_head(model):
    """The transformer that produces hidden states, and the output layer's weight, of a (LoRA-wrapped) model."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings().weight


def hidden_states(model, input_ids: torch.Tensor) -> torch.Tensor:
    decoder, _ = decoder_and_head(model)
    # Right padding and no attention mask: a causal model reads left to right, so real tokens never see the padding
    # after them. (A padding mask produced NaN on fully masked rows in earlier runs on these GPUs.)
    return decoder(input_ids=input_ids, use_cache=False).last_hidden_state


def pad(rows: list[list[int]], value: int) -> torch.Tensor:
    width = max(len(r) for r in rows)
    return torch.tensor([r + [value] * (width - len(r)) for r in rows], dtype=torch.long)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the ROC curve: the chance that a random positive scores above a random negative (ties count half)."""
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return None
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):  # average ranks over ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


# ---------------------------------------------------------------- decision models, the letters way

def render_state(state) -> str:
    """{"message": "hi"} reads as "message: hi"; other values as compact JSON."""
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}" for k, v in state.items())
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def options(question: dict) -> list[tuple[object, str]]:
    """(answer value, text) for each option of a question, in the question's own order."""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "noul":
        c = criteria or {}
        return [(True, f"yes: {c['true']}" if c.get("true") else "yes"),
                (False, f"no: {c['false']}" if c.get("false") else "no")]
    if kind == "choice":
        return [(key, f"{key}: {text}" if isinstance(text, str) and text else str(key)) for key, text in criteria.items()]
    return [(i, level if isinstance(level, str) else json.dumps(level, ensure_ascii=False)) for i, level in enumerate(criteria)]


def target(question: dict, answer, opts: list[tuple[object, str]]) -> list[float]:
    """The answer as a probability per option: one-hot, or the soft label's probabilities when it has them."""
    keys = [k for k, _ in opts]
    if isinstance(answer, dict) and isinstance(answer.get("probabilities"), dict):
        probs = {str(k).lower(): float(v) for k, v in answer["probabilities"].items()}
        vec = [probs.get(str(k).lower(), 0.0) for k in keys]
        if sum(vec) > 0:
            return [v / sum(vec) for v in vec]
    label = answer.get("label") if isinstance(answer, dict) else answer
    if question["type"] == "noul":
        label = label is True or label == "true"
    vec = [0.0] * len(keys)
    vec[keys.index(label)] = 1.0
    return vec


def add_answers(questions: dict, name: str, kind: str, lists: dict) -> None:
    """Add answers to their question's lists, kept by (name, type): a name asked as two types (rare) is two
    questions."""
    entry = questions.setdefault((name, kind), {key: [] for key in lists})
    for key, values in lists.items():
        entry[key].extend(values)


def note(questions: dict, name: str, kind: str, **values) -> None:
    """One answer: for yes/no, the chance of yes and the gold answer; otherwise the option picked and the gold one."""
    add_answers(questions, name, kind, {key: [value] for key, value in values.items()})


class Decisions:
    """What the two decision methods share: how their answers are combined and scored."""

    kind = "decision"
    false_alarm_rate = 0.01  # for the catch rate: the share of no cases that may be flagged (a run setting)

    @staticmethod
    def combine(parts: list[dict]) -> dict:
        """Several cards' outputs as one."""
        out = {"loss": [], "correct": [], "confidence": [], "questions": {}}
        for part in parts:
            for key in ("loss", "correct", "confidence"):
                out[key] += part[key]
            for (name, kind), answers in part["questions"].items():
                add_answers(out["questions"], name, kind, answers)
        return out

    def metrics(self, out: dict, rows: int) -> dict:
        return decision_metrics(out, rows, self.false_alarm_rate)


class NoHead:
    """Tasks that train only the model itself (Clef also trains its head)."""

    head = None

    def parameters(self) -> list:
        return []

    def save(self, folder) -> None:
        pass

    def load(self, folder) -> None:
        pass


@dataclass
class Item:
    row: int
    name: str


class DecisionLetters(Decisions, NoHead):
    def __init__(self, tokenizer, max_length: int, seed: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.letter_ids = []
        for letter in LETTERS:
            ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("This tokenizer splits the option letters into several pieces, so the letters method "
                                 "can't read answers from it. Try another model.")
            self.letter_ids.append(ids[0])

    def items(self, df) -> tuple[list[Item], dict]:
        """One item per (row, question); the parsed rows are kept for building prompts."""
        parsed = {"state": [], "questions": [], "gold": []}
        items = []
        for i, (state, questions, gold) in enumerate(zip(df["state"], df["questions"], df["gold"])):
            q = _value(questions)
            parsed["state"].append(render_state(_value(state)))
            parsed["questions"].append(q)
            parsed["gold"].append(_value(gold))
            for name, question in q.items():
                if len(options(question)) > len(LETTERS):
                    raise ValueError(f"Question '{name}' has more than {len(LETTERS)} options, which the letters "
                                     "method can't show. The Clef head (step 6) can.")
                items.append(Item(i, name))
        return items, parsed

    def lengths(self, items: list[Item], parsed: dict) -> np.ndarray:
        """Rough token counts (about 3.5 characters per token), only used to group similar lengths in a batch."""
        return np.array([min(self.max_length, (len(parsed["state"][it.row])
                                               + len(json.dumps(parsed["questions"][it.row][it.name]))) // 3.5 + 8)
                         for it in items])

    def _encode(self, state: str, question: dict, name: str, order: list[int]) -> tuple[list[int], list[str]]:
        opts = options(question)
        lines = "\n".join(f"{LETTERS[k]}. {opts[o][1]}" for k, o in enumerate(order))
        tail = self.tok.encode(f"\n\nQuestion: {question.get('instructions') or name}\n{lines}\nAnswer:",
                               add_special_tokens=False)
        room = max(16, self.max_length - len(tail))
        head = self.tok.encode(state, add_special_tokens=False)[:room]
        return head + tail, [opts[o][1] for o in order]

    def collate(self, items: list[Item], parsed: dict, shuffle: bool, index_base: int = 0,
                reverse: bool = False) -> dict:
        ids, targets, n_options = [], [], []
        for n, it in enumerate(items):
            question = parsed["questions"][it.row][it.name]
            opts = options(question)
            order = list(range(len(opts)))
            if shuffle:  # the same row always gets the same order, so a resumed run sees the same data
                random.Random(self.seed * 1_000_003 + it.row * 7919 + zlib.crc32(it.name.encode())).shuffle(order)
            elif reverse:
                order.reverse()
            encoded, _ = self._encode(parsed["state"][it.row], question, it.name, order)
            vec = target(question, parsed["gold"][it.row][it.name], opts)
            ids.append(encoded)
            targets.append([vec[o] for o in order])
            n_options.append(len(opts))
        width = max(n_options)
        return {
            "input_ids": pad(ids, self.tok.pad_token_id or 0),
            "lengths": torch.tensor([len(x) for x in ids]),
            "targets": torch.tensor([t + [0.0] * (width - len(t)) for t in targets]),
            "valid": torch.tensor([[k < n for k in range(width)] for n in n_options]),
            "tokens": sum(len(x) for x in ids),
        }

    def logits(self, model, batch: dict) -> torch.Tensor:
        device = next(model.parameters()).device
        ids = batch["input_ids"].to(device)
        hidden = hidden_states(model, ids)
        last = hidden[torch.arange(len(ids), device=device), batch["lengths"].to(device) - 1]
        _, head = decoder_and_head(model)
        width = batch["valid"].shape[1]
        weight = head[torch.tensor(self.letter_ids[:width], device=device)]
        logits = last.float() @ weight.float().T
        return logits.masked_fill(~batch["valid"].to(device), float("-inf"))

    def loss(self, model, batch: dict) -> torch.Tensor:
        logits = self.logits(model, batch)
        targets = batch["targets"].to(logits.device)
        logp = torch.log_softmax(logits, dim=-1).masked_fill(~batch["valid"].to(logits.device), 0.0)
        return -(targets * logp).sum(-1).mean()

    @torch.no_grad()
    def _probabilities(self, model, items: list[Item], parsed: dict, autocast, reverse: bool) -> torch.Tensor:
        batch = self.collate(items, parsed, shuffle=False, reverse=reverse)
        with autocast():
            probs = torch.softmax(self.logits(model, batch), dim=-1).float().cpu()
        if reverse:  # back to each question's own option order
            counts = batch["valid"].sum(-1).tolist()
            probs = torch.stack([torch.cat([p[:n].flip(0), p[n:]]) for p, n in zip(probs, counts)])
        return probs

    def evaluate(self, model, items: list[Item], parsed: dict, batch_size: int, autocast) -> dict:
        """Scores on held-out rows (see metrics)."""
        return self.metrics(self.outputs(model, items, parsed, batch_size, autocast), len(items))

    @torch.no_grad()
    def outputs(self, model, items: list[Item], parsed: dict, batch_size: int, autocast) -> dict:
        """The model's answer to each question, ready to be combined with other cards' shares and scored. Every
        question is asked in its own option order and in the reverse order, and the two answers are averaged, so a
        model that favours a letter isn't scored on that."""
        losses, correct, confidence, questions = [], [], [], {}
        order = sorted(range(len(items)), key=lambda i: len(parsed["state"][items[i].row]))
        for start in range(0, len(order), batch_size):
            chunk = [items[i] for i in order[start:start + batch_size]]
            probs = (self._probabilities(model, chunk, parsed, autocast, False)
                     + self._probabilities(model, chunk, parsed, autocast, True)) / 2
            targets = self.collate(chunk, parsed, shuffle=False)
            valid, targets = targets["valid"], targets["targets"]
            logp = torch.log(probs.clamp_min(1e-9)).masked_fill(~valid, 0.0)
            losses += (-(targets * logp).sum(-1)).tolist()
            picked = probs.masked_fill(~valid, -1)
            correct += (picked.argmax(-1) == targets.argmax(-1)).tolist()
            confidence += picked.max(-1).values.tolist()
            for it, p, t, k in zip(chunk, probs, targets, picked.argmax(-1).tolist()):
                question = parsed["questions"][it.row][it.name]
                opts, gold = options(question), int(t.argmax())
                if question["type"] == "noul":  # option 0 is "yes" in the question's own order
                    note(questions, it.name, "noul", yes=float(p[0]), gold=bool(t[0] >= 0.5))
                elif question["type"] == "choice":
                    note(questions, it.name, "choice", picked=str(opts[k][0]), gold=str(opts[gold][0]))
                else:  # a score's options are its levels, in order
                    note(questions, it.name, "score", picked=k, gold=gold, levels=len(opts))
        return {"loss": losses, "correct": correct, "confidence": confidence, "questions": questions}


def decision_metrics(out: dict, rows: int, false_alarm_rate: float) -> dict:
    """
    Over all questions: val_loss, accuracy, and ece (calibration error: how far "80% sure" is from being right 80% of
    the time; lower is better).

    Each question by its type, in "questions", with how many rows asked it and how often each answer was the gold one:
    - yes/no: auroc; catch_rate, the share of yes caught when at most false_alarm_rate of the no cases are flagged; and
      false_alarms, the share of no cases flagged at a 0.5 cut-off
    - choice: accuracy, and macro_f1 (F1 per answer, averaged, so a rare answer counts as much as a common one)
    - score: accuracy (the exact level), and mae (how many levels off, on average)

    auroc, catch_rate, false_alarms, macro_f1 and mae average them over the questions of their type. overall averages
    a 0-to-1 quality over every question: its AUROC, its macro-F1, or 1 - its MAE / (levels - 1).
    """
    correct = np.array(out["correct"], dtype=bool)
    result = {"val_loss": float(np.mean(out["loss"])), "accuracy": float(correct.mean()),
              "ece": calibration_error(np.array(out["confidence"]), correct), "rows": rows}
    scored, quality = {}, []
    types_of = Counter(name for name, _ in out["questions"])
    for (name, kind), answers in sorted(out["questions"].items()):
        label = name if types_of[name] == 1 else f"{name} ({kind})"
        scored[label], value = SCORE_BY_TYPE[kind](answers, false_alarm_rate)
        if value is not None:
            quality.append(value)
    for kind, key in (("noul", "auroc"), ("noul", "catch_rate"), ("noul", "false_alarms"), ("choice", "macro_f1"),
                      ("score", "mae")):
        values = [s[key] for s in scored.values() if s["type"] == kind and key in s]
        if values:
            result[key] = float(np.mean(values))
    if quality:
        result["overall"] = float(np.mean(quality))
    result["questions"] = scored
    return result


def _counts(values: list, by_count: bool) -> dict:
    """How often each answer is the gold one: the most common first, or in order (a score's levels)."""
    found = Counter(values)
    keys = [k for k, _ in found.most_common()] if by_count else sorted(found)
    return {str(k): found[k] for k in keys}


def _score_yes_no(answers: dict, false_alarm_rate: float) -> tuple[dict, float | None]:
    p, y = np.array(answers["yes"], dtype=float), np.array(answers["gold"], dtype=bool)
    scored = {"type": "noul", "rows": len(y), "answers": {"yes": int(y.sum()), "no": int((~y).sum())}}
    value = auroc(p, y)
    if value is not None:  # needs both answers among the gold ones
        scored.update(auroc=value, catch_rate=catch_rate(p, y, false_alarm_rate),
                      false_alarms=float((p[~y] >= 0.5).mean()))
    return scored, value


def _score_choice(answers: dict, false_alarm_rate: float) -> tuple[dict, float]:
    picked, gold = answers["picked"], answers["gold"]
    f1 = macro_f1(picked, gold)
    return {"type": "choice", "rows": len(gold), "answers": _counts(gold, by_count=True),
            "accuracy": float(np.mean([p == g for p, g in zip(picked, gold)])), "macro_f1": f1}, f1


def _score_level(answers: dict, false_alarm_rate: float) -> tuple[dict, float | None]:
    picked, gold, levels = (np.array(answers[k], dtype=float) for k in ("picked", "gold", "levels"))
    off = np.abs(picked - gold)
    spread = levels > 1  # a question with a single level can't be off
    quality = float(1 - (off[spread] / (levels[spread] - 1)).mean()) if spread.any() else None
    return {"type": "score", "rows": len(gold), "answers": _counts([int(g) for g in gold], by_count=False),
            "accuracy": float((off == 0).mean()), "mae": float(off.mean())}, quality


SCORE_BY_TYPE = {"noul": _score_yes_no, "choice": _score_choice, "score": _score_level}


def macro_f1(picked: list, gold: list) -> float:
    """F1 for each answer that was the gold one or was picked, averaged: each counts the same, however rare. F1 is
    2·hits / (2·hits + wrongly picked + missed)."""
    picked, gold = np.array(picked, dtype=object), np.array(gold, dtype=object)
    scores = []
    for answer in set(gold.tolist()) | set(picked.tolist()):
        hits = int(((picked == answer) & (gold == answer)).sum())
        wrong = int(((picked == answer) & (gold != answer)).sum())
        missed = int(((picked != answer) & (gold == answer)).sum())
        scores.append(2 * hits / (2 * hits + wrong + missed))
    return float(np.mean(scores))


def catch_rate(scores: np.ndarray, labels: np.ndarray, false_alarm_rate: float) -> float:
    """Share of positives scored above the cut-off that flags at most `false_alarm_rate` of the negatives."""
    negatives = np.sort(scores[~labels])[::-1]
    allowed = int(np.floor(false_alarm_rate * len(negatives)))  # how many negatives may be flagged
    cutoff = negatives[allowed] if allowed < len(negatives) else -np.inf
    return float((scores[labels] > cutoff).mean())


def calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: the gap between how sure the model is and how often it's right, averaged over
    10 confidence bins weighted by how many answers fall in each."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            total += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(total)


# ---------------------------------------------------------------- language models

class LanguageModel(NoHead):
    kind = "lm"

    def __init__(self, tokenizer, fmt: str, max_length: int):
        self.tok = tokenizer
        self.fmt = fmt
        self.max_length = max_length
        if fmt == "chat" and not getattr(tokenizer, "chat_template", None):
            raise ValueError("This model has no chat template, so it can't be trained on chat data. Use prompt and "
                             "completion or plain text, or pick a model with a chat template.")
        self.eos = tokenizer.eos_token_id
        self.end_markers = sorted((t for t in tokenizer.all_special_tokens if t), key=len, reverse=True)

    def items(self, df) -> tuple[list[int], dict]:
        columns = {"chat": ["messages"], "pairs": ["prompt", "completion"], "text": ["text"]}[self.fmt]
        parsed = {c: [(_value(v) if c == "messages" else v) for v in df[c]] for c in columns}
        return list(range(len(df))), parsed

    def lengths(self, items: list[int], parsed: dict) -> np.ndarray:
        def chars(i):
            if self.fmt == "chat":
                return sum(len(str(m.get("content", ""))) + 16 for m in parsed["messages"][i])
            if self.fmt == "pairs":
                return len(parsed["prompt"][i]) + len(parsed["completion"][i])
            return len(parsed["text"][i])
        return np.array([min(self.max_length, chars(i) // 3.5 + 4) for i in items])

    def _chat(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """Token ids of the whole conversation, and labels that are -100 everywhere but the assistant's replies."""
        text = self.tok.apply_chat_template(messages, tokenize=False)
        spans, pos = [], 0
        for m in messages:
            if m.get("role") != "assistant":
                continue
            content = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
            content = content.strip()
            start = text.find(content, pos) if content else -1
            if start < 0:
                spans = None
                break
            end = start + len(content)
            marker = next((t for t in self.end_markers if text.startswith(t, end)), None)
            end += len(marker) if marker else 0  # learn to end the turn, too
            spans.append((start, end))
            pos = end
        if spans is None:  # a template that rewrites replies: learn the last reply only
            prefix = self.tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            spans = [(len(prefix), len(text))] if text.startswith(prefix) else [(0, len(text))]
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"][: self.max_length]
        labels = [tid if any(s <= a and b <= e and b > a for s, e in spans) else -100
                  for tid, (a, b) in zip(ids, enc["offset_mapping"])]
        return ids, labels

    def example(self, i: int, parsed: dict) -> tuple[list[int], list[int]]:
        if self.fmt == "chat":
            return self._chat(parsed["messages"][i])
        if self.fmt == "pairs":
            prompt = self.tok.encode(parsed["prompt"][i], add_special_tokens=False)
            completion = self.tok.encode(parsed["completion"][i], add_special_tokens=False) + [self.eos]
            ids = (prompt + completion)[: self.max_length]
            labels = ([-100] * len(prompt) + completion)[: self.max_length]
            return ids, labels
        ids = self.tok.encode(parsed["text"][i], add_special_tokens=False)[: self.max_length - 1] + [self.eos]
        return ids, list(ids)

    def collate(self, items: list[int], parsed: dict, shuffle: bool = False, index_base: int = 0) -> dict:
        pairs = [self.example(i, parsed) for i in items]
        return {
            "input_ids": pad([p[0] for p in pairs], self.tok.pad_token_id or 0),
            "labels": pad([p[1] for p in pairs], -100),
            "tokens": sum(len(p[0]) for p in pairs),
        }

    def _sum_and_count(self, model, batch: dict) -> tuple[torch.Tensor, int]:
        device = next(model.parameters()).device
        hidden = hidden_states(model, batch["input_ids"].to(device))
        labels = batch["labels"].to(device)[:, 1:]
        mask = labels != -100
        h, y = hidden[:, :-1][mask], labels[mask]
        _, head = decoder_and_head(model)
        total = hidden.new_zeros((), dtype=torch.float32)

        def chunk_loss(h_part, y_part):
            return F.cross_entropy((h_part @ head.T).float(), y_part, reduction="sum")

        for start in range(0, len(y), LOSS_CHUNK):  # recomputed in the backward pass, so only one chunk's scores
            total = total + checkpoint(chunk_loss, h[start:start + LOSS_CHUNK], y[start:start + LOSS_CHUNK],
                                       use_reentrant=False)
        return total, int(len(y))

    def loss(self, model, batch: dict) -> torch.Tensor:
        total, count = self._sum_and_count(model, batch)
        return total / max(count, 1)

    def evaluate(self, model, items: list[int], parsed: dict, batch_size: int, autocast) -> dict:
        return self.metrics(self.outputs(model, items, parsed, batch_size, autocast), len(items))

    @torch.no_grad()
    def outputs(self, model, items: list[int], parsed: dict, batch_size: int, autocast) -> dict:
        total, count = 0.0, 0
        for start in range(0, len(items), batch_size):
            batch = self.collate(items[start:start + batch_size], parsed)
            with autocast():
                s, n = self._sum_and_count(model, batch)
            total += float(s)
            count += n
        return {"total": total, "count": count}

    @staticmethod
    def combine(parts: list[dict]) -> dict:
        return {"total": sum(p["total"] for p in parts), "count": sum(p["count"] for p in parts)}

    @staticmethod
    def metrics(out: dict, rows: int) -> dict:
        loss = out["total"] / max(out["count"], 1)
        return {"val_loss": loss, "perplexity": float(math.exp(min(loss, 50))), "rows": rows}


# ---------------------------------------------------------------- decision models, the Clef way

def clef_head_config(hidden_size: int) -> dict:
    """A fresh head's size, as in Cloudflare's releases: 1024 wide (512 on models narrower than 3072), 2 evidence-routing
    layers and 4 decoder layers, an attention head per 64 dimensions, and a feed-forward 4 times the width."""
    width = 1024 if hidden_size >= 3072 else 512
    return {"hidden_size": hidden_size, "width": width, "routing_layers": 2, "layers": 4, "heads": width // 64,
            "feedforward": 4 * width, "dropout": 0.0}


def clef_target(question: dict, answer, option_ids: tuple[str, ...]) -> list[float]:
    """The answer as a probability per Clef option: noul's options are "true" and "false", choice's are its keys in
    sorted order, and score's are the levels "0", "1", ..."""
    kind = question["type"]
    keys = [True, False] if kind == "noul" else [int(o) for o in option_ids] if kind == "score" else list(option_ids)
    return target(question, answer, [(k, "") for k in keys])


class DecisionClef(Decisions):
    """Each row (its state and every question with its allowed options) becomes one prompt, built by Clef's own
    encode_record. The head reads the model's hidden states over each question and option, and the option words'
    output embeddings, and gives one score per option. The loss is cross-entropy against the gold answer, averaged over
    questions. An item is a row, since one pass answers all of its questions."""

    def __init__(self, tokenizer, max_length: int, head: torch.nn.Module, config: dict):
        from .vendor.clef import joint_schema_model as clef

        self.clef = clef
        self.tok, self.max_length, self.head, self.config = tokenizer, max_length, head, config

    def parameters(self) -> list:
        return list(self.head.parameters())

    def save(self, folder) -> None:
        """The head as Clef releases ship it: joint_head_config.json and joint_head.safetensors."""
        from pathlib import Path

        from safetensors.torch import save_file

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "joint_head_config.json").write_text(json.dumps(self.config, indent=1))
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
                  str(folder / "joint_head.safetensors"))

    def load(self, folder) -> None:
        from pathlib import Path

        from safetensors.torch import load_file

        device = next(self.head.parameters()).device
        state = load_file(str(Path(folder) / "joint_head.safetensors"), device=str(device))
        self.head.load_state_dict({k: v.to(torch.float32) for k, v in state.items()}, strict=True)

    def items(self, df) -> tuple[list[int], dict]:
        parsed = {"state": [], "questions": [], "gold": []}
        for state, questions, gold in zip(df["state"], df["questions"], df["gold"]):
            parsed["state"].append(_value(state))
            parsed["questions"].append(_value(questions))
            parsed["gold"].append(_value(gold))
        self._check_room(parsed["questions"])
        return list(range(len(parsed["state"]))), parsed

    def _check_room(self, all_questions: list[dict]) -> None:
        """Every question goes in the prompt, so they must fit in max length: say so before training starts. Rows
        usually share a few sets of questions, so each set is tried once (up to 2,000 of them)."""
        seen = set()
        for questions in all_questions:
            key = json.dumps(questions, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            try:
                self.clef.encode_record(self.tok, {"state": "", "questions": questions}, max_length=self.max_length)
            except ValueError as error:
                raise ValueError(f"Max length {self.max_length} is too short for the Clef prompt, which holds every "
                                 f"question of a row ({error}). Raise Max length under Hyperparameters, Advanced.") from None
            if len(seen) >= 2000:
                return

    def lengths(self, items: list[int], parsed: dict) -> np.ndarray:
        """Rough token counts (about 3.5 characters per token), only used to group similar lengths in a batch."""
        return np.array([min(self.max_length, (len(self.clef.render(parsed["state"][i]))
                                               + len(json.dumps(parsed["questions"][i]))) // 3.5 + 64) for i in items])

    def collate(self, items: list[int], parsed: dict, shuffle: bool = False) -> dict:
        records, targets = [], []
        for i in items:
            questions = parsed["questions"][i]
            record = self.clef.encode_record(self.tok, {"state": parsed["state"][i], "questions": questions},
                                             max_length=self.max_length)
            records.append(record)
            targets.append([torch.tensor(clef_target(questions[q.question_id], parsed["gold"][i][q.question_id],
                                                     q.option_ids)) for q in record.questions])
        ids = [list(r.input_ids) for r in records]
        return {"input_ids": pad(ids, self.tok.pad_token_id or 0), "lengths": torch.tensor([len(x) for x in ids]),
                "records": records, "targets": targets, "tokens": sum(len(x) for x in ids)}

    def logits(self, model, batch: dict) -> list[list[torch.Tensor]]:
        """One tensor of option scores per question, per row."""
        device = next(model.parameters()).device
        ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        mask = (torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]).long()
        _, output_embeddings = decoder_and_head(model)
        return self.head(hidden_states(model, ids), ids, mask, batch["records"], output_embeddings)

    def loss(self, model, batch: dict) -> torch.Tensor:
        total, count = None, 0
        for record_logits, record_targets in zip(self.logits(model, batch), batch["targets"]):
            for scores, t in zip(record_logits, record_targets):
                term = -(t.to(scores.device) * torch.log_softmax(scores.float(), dim=-1)).sum()
                total = term if total is None else total + term
                count += 1
        return total / count

    def evaluate(self, model, items: list[int], parsed: dict, batch_size: int, autocast) -> dict:
        return self.metrics(self.outputs(model, items, parsed, batch_size, autocast), len(items))

    @torch.no_grad()
    def outputs(self, model, items: list[int], parsed: dict, batch_size: int, autocast) -> dict:
        losses, correct, confidence, questions = [], [], [], {}
        kinds = {number: name for name, number in self.clef.QUESTION_TYPES.items()}
        order = sorted(items, key=lambda i: len(self.clef.render(parsed["state"][i])))
        for start in range(0, len(order), batch_size):
            batch = self.collate(order[start:start + batch_size], parsed)
            with autocast():
                logits = self.logits(model, batch)
            for record, record_logits, record_targets in zip(batch["records"], logits, batch["targets"]):
                for question, scores, t in zip(record.questions, record_logits, record_targets):
                    p = torch.softmax(scores.float(), dim=-1).cpu()
                    losses.append(float(-(t * torch.log(p.clamp_min(1e-9))).sum()))
                    correct.append(bool(p.argmax() == t.argmax()))
                    confidence.append(float(p.max()))
                    kind, ids = kinds[question.question_type], question.option_ids
                    k, gold = int(p.argmax()), int(t.argmax())
                    if kind == "noul":  # option 0 is "true"
                        note(questions, question.question_id, kind, yes=float(p[0]), gold=bool(t[0] >= 0.5))
                    elif kind == "choice":
                        note(questions, question.question_id, kind, picked=ids[k], gold=ids[gold])
                    else:  # a score's options are its levels, "0", "1", ...
                        note(questions, question.question_id, kind, picked=int(ids[k]), gold=int(ids[gold]),
                             levels=len(ids))
        return {"loss": losses, "correct": correct, "confidence": confidence, "questions": questions}

    def metrics(self, out: dict, rows: int) -> dict:
        # rows: the questions scored, like the letters way (a row with several questions counts several times)
        return decision_metrics(out, len(out["loss"]), self.false_alarm_rate)


def length_grouped_batches(lengths: np.ndarray, batch_size: int, rng: random.Random, chunk: int = 4096) -> list[list[int]]:
    """Shuffle, sort within big chunks by length, cut batches of batch_size rows, then shuffle the batches.
    Rows of similar length share a batch, so little compute is spent on padding."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start:start + chunk], key=lambda i: lengths[i], reverse=True)
        batches += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches
