"""What the model learns from a row, and how its loss and validation scores are computed.

Two tasks:
- DecisionLetters: a decision model the letters way. Each (state, question) becomes a prompt that lists the options as
  A, B, C... and ends at "Answer:". The model's next-token scores for the option letters are its answer, so nothing is
  generated. Option order is shuffled per row (seeded) while training, so no letter is favoured. One question per pass.
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


@dataclass
class Item:
    row: int
    name: str


class DecisionLetters:
    kind = "decision"

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

    @torch.no_grad()
    def evaluate(self, model, items: list[Item], parsed: dict, batch_size: int, autocast) -> dict:
        """Scores on held-out rows. Every question is asked in its own option order and in the reverse order, and the
        two answers are averaged, so a model that favours a letter isn't scored on that.

        - val_loss, accuracy: over all questions
        - ece: calibration error, how far "80% sure" is from being right 80% of the time (lower is better)
        - per yes/no question: auroc_<name>, catch_<name> (share of yes caught when at most 1% of no is flagged),
          and false_alarms_<name> (share of no flagged at a 0.5 cut-off); auroc, catch_rate, false_alarms average them
        """
        losses, correct, confidence, yes = [], [], [], {}
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
            for it, p, t in zip(chunk, probs, targets):
                if parsed["questions"][it.row][it.name]["type"] == "noul":
                    yes.setdefault(it.name, ([], []))
                    yes[it.name][0].append(float(p[0]))  # option 0 is "yes" in the question's own order
                    yes[it.name][1].append(bool(t[0] >= 0.5))
        out = {"val_loss": float(np.mean(losses)), "accuracy": float(np.mean(correct)),
               "ece": calibration_error(np.array(confidence), np.array(correct, dtype=bool)), "rows": len(items)}
        per = {"auroc": [], "catch": [], "false_alarms": []}
        for name, (p, y) in yes.items():
            p, y = np.array(p), np.array(y)
            value = auroc(p, y)
            if value is None:
                continue
            caught = catch_rate(p, y, 0.01)
            alarms = float((p[~y] >= 0.5).mean())
            out[f"auroc_{name}"], out[f"catch_{name}"], out[f"false_alarms_{name}"] = value, caught, alarms
            per["auroc"].append(value)
            per["catch"].append(caught)
            per["false_alarms"].append(alarms)
        if per["auroc"]:
            out["auroc"] = float(np.mean(per["auroc"]))
            out["catch_rate"] = float(np.mean(per["catch"]))
            out["false_alarms"] = float(np.mean(per["false_alarms"]))
        return out


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

class LanguageModel:
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

    @torch.no_grad()
    def evaluate(self, model, items: list[int], parsed: dict, batch_size: int, autocast) -> dict:
        total, count = 0.0, 0
        for start in range(0, len(items), batch_size):
            batch = self.collate(items[start:start + batch_size], parsed)
            with autocast():
                s, n = self._sum_and_count(model, batch)
            total += float(s)
            count += n
        loss = total / max(count, 1)
        return {"val_loss": loss, "perplexity": float(math.exp(min(loss, 50)))}


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
