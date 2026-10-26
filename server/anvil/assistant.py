"""The pipeline assistant: you describe the data you want, and a model builds the pipeline with you.

The assistant is any OpenAI-compatible model that takes tools (function calling), set up once in Settings: an
address, a model and a key, like a Model step's. Each message you send starts a turn, in a process of its own like a
run (python -m anvil.assistant <pipeline folder>/assistant <turn>), so it outlives a restart of the server. The turn
asks the model, carries out the tools it calls on the pipeline (building it, changing steps, previewing, reading
datasets), hands back what came of them, and asks again until the model answers without calling one.

It's a conversation, free form: the assistant talks a new pipeline through first (it asks what it needs to know and
proposes the steps) and builds it when you say so, while a clear change you ask for it just makes. It only previews:
a full run is yours to start. While a turn works, the pipeline can't be edited or run from the screen, and every turn
that changed it can be undone. A server that holds one model at a time can't answer
the assistant while a run asks it for another, so a turn waits while a run (or preview) uses the assistant's address
with a different model: you can stop that run, or have it ask anyway.

On disk:
- ANVIL_DATA/assistant/settings.json   the model: address, model, temperature, max tokens, thinking, reasoning
                                       effort, context, timeout
- ANVIL_DATA/assistant/secrets.json    its API key (0600), never sent back to the screen. A Model step at the same
                                       address that has no key of its own uses it too (anvil.pipelines.key_for)
- <pipeline>/assistant/chat.json       the conversation as the model sees it, with what the screen shows for each
- <pipeline>/assistant/turn.json       the newest turn: its state (starting, working, waiting, done, failed,
                                       stopped), what it's doing, what it waits for, the error, its process
- <pipeline>/assistant/before-<n>.json the pipeline as it was before turn n changed it, for Undo
- <pipeline>/assistant/stop.json, go.json   Stop, and Ask anyway
- <pipeline>/assistant/log.txt         what went wrong, if anything
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import config, folders, llm, pipelines, templates

DEFAULTS = {"base_url": "http://host.docker.internal:8000/v1", "model": "", "temperature": 0.3, "max_tokens": 4096,
            "thinking": "default", "reasoning_effort": "default", "context": 32768, "timeout": 600}
# reasoning_effort: the levels models take differ (OpenAI's low, medium and high; Qwen3.8 on radiance low, medium and
# xhigh), and a level a model doesn't take is refused, its reply saying which it does. none turns thinking off.
EFFORTS = ("default", "none", "low", "medium", "high", "xhigh")
WORKING = {"starting", "working", "waiting"}
MESSAGE_CHARS = 8000
MAX_CALLS = 24  # model calls in one turn
MAX_PREVIEWS = 5  # previews in one turn
PREVIEW_WAIT_S = 900
WAIT_POLL_S = 3
TEXT_CHARS = 400  # a column's value in a tool's result
PROMPT_CHARS = 1200
ASKED_SHOWN = 6  # prompts and replies a preview's result shows
CHARS_PER_TOKEN = 3.2  # a cautious guess, for keeping the conversation inside the model's context
NODE_WIDTH, ROW_GAP, COLUMN_GAP = 248, 150, 40  # the canvas's (web/src/pipeline/graph.ts tidy)

_processes: dict[str, subprocess.Popen] = {}


class Stopped(Exception):
    """Stop was pressed."""


# ---------------------------------------------------------------- the assistant's model

def settings_dir() -> Path:
    return config.settings.data / "assistant"


def key() -> str | None:
    return pipelines._read(settings_dir() / "secrets.json", {}).get("key") or None


def settings() -> dict:
    saved = pipelines._read(settings_dir() / "settings.json", {}) or {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}, "has_key": bool(key())}


def save_settings(body: dict) -> dict:
    """Save the assistant's model. ValueError says what's wrong with a setting."""
    current = {k: v for k, v in settings().items() if k in DEFAULTS}
    given = {k: v for k, v in body.items() if k in DEFAULTS}
    if "base_url" in given:
        url = llm.base_url(str(given["base_url"]))
        problem = llm.check_url(url)
        if problem:
            raise ValueError(problem)
        given["base_url"] = url
    if "model" in given:
        given["model"] = str(given["model"] or "").strip()[:200]
    if "thinking" in given and given["thinking"] not in ("default", "on", "off"):
        raise ValueError("Thinking is default, on or off.")
    if "reasoning_effort" in given and given["reasoning_effort"] not in EFFORTS:
        raise ValueError(f"Reasoning effort is one of {', '.join(EFFORTS)}.")
    for field, low, high, what in (("temperature", 0, 2, "Temperature"), ("max_tokens", 256, 131072, "Max tokens"),
                                   ("context", 4096, 2_000_000, "Context"), ("timeout", 30, 3600, "The timeout")):
        if field not in given or (field == "temperature" and given[field] is None):
            continue
        value = given[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high \
                or (field != "temperature" and value != int(value)):
            raise ValueError(f"{what} goes from {low:,} to {high:,}.")
        given[field] = value if field == "temperature" else int(value)
    settings_dir().mkdir(parents=True, exist_ok=True)
    pipelines._write(settings_dir() / "settings.json", {**current, **given})
    return settings()


def set_key(value: str | None) -> None:
    settings_dir().mkdir(parents=True, exist_ok=True)
    pipelines._write(settings_dir() / "secrets.json", {"key": value} if value else {}, private=True)


def key_at(url: str) -> str | None:
    """The assistant's key, when url is the assistant's address: Model steps there can use it."""
    saved = pipelines._read(settings_dir() / "settings.json", {}) or {}
    mine = llm.base_url(str(saved.get("base_url") or DEFAULTS["base_url"]))
    return key() if url and llm.base_url(url) == mine else None


def models(url: str | None = None) -> dict:
    """The models an address serves (the assistant's own, by default), asked with the assistant's key."""
    base, found = llm.list_models(url or settings()["base_url"], key())
    return {"base_url": base, "models": found}


# ---------------------------------------------------------------- a pipeline's conversation

def home(pipeline_id: str) -> Path:
    return pipelines.folder(pipeline_id) / "assistant"


def _chat(where: Path) -> dict:
    return pipelines._read(where / "chat.json", None) or {"turns": 0, "messages": []}


def _alive(where: Path, pid: int | None) -> bool:
    proc = _processes.get(str(where))
    if proc is not None:
        return proc.poll() is None
    if not pid:
        return False
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return b"anvil.assistant" in args and str(where).encode() in args


def _turn(where: Path) -> dict:
    """The newest turn, corrected when its process ended without saying so (a crash, or anvil restarting)."""
    current = pipelines._read(where / "turn.json", {}) or {}
    if current.get("state") in WORKING and not _alive(where, current.get("pid")):
        current.update(state="failed", doing=None, waiting_for=None, error=current.get("error") or (
            "The assistant stopped without finishing (anvil may have restarted). Send your message again."))
        pipelines._write(where / "turn.json", current)
    return current


def working(pipeline_id: str) -> bool:
    try:
        where = home(pipeline_id)
    except FileNotFoundError:
        return False
    return _turn(where).get("state") in WORKING


def _undoable(where: Path) -> int | None:
    found = [int(m.group(1)) for p in where.glob("before-*.json") if (m := re.fullmatch(r"before-(\d+)\.json", p.name))]
    return max(found) if found else None


def conversation(pipeline_id: str) -> dict:
    """The conversation as the screen shows it, the newest turn, and whether Undo has something to undo."""
    where = home(pipeline_id)
    shown = []
    for entry in _chat(where)["messages"]:
        role, at = entry["role"], entry.get("at")
        if role == "user":
            shown.append({"role": "user", "text": entry.get("text", ""), "at": at})
        elif role == "assistant" and entry.get("content"):
            shown.append({"role": "assistant", "text": entry["content"], "at": at})
        elif role == "tool":
            shown.append({"role": "action", "tool": entry.get("tool"), "text": entry.get("summary", ""),
                          "ok": entry.get("ok", True), "at": at})
        elif role == "note":
            shown.append({"role": "note", "text": entry.get("text", ""), "at": at})
    turn = {k: v for k, v in _turn(where).items() if k != "pid"}
    s = settings()
    return {"messages": shown, "turn": turn or None, "undo": None if turn.get("state") in WORKING else _undoable(where),
            "model": s["model"], "ready": bool(s["model"])}


def describe(data: dict) -> dict:
    """A pipeline the way the assistant reads it: its steps, arrows and what stops it from running."""
    return {
        "name": data["name"], "about": data.get("about", ""),
        "steps": [{"id": n["id"], "type": n["type"], **({"note": n["note"]} if n.get("note") else {}),
                   "config": n["config"]} for n in data["nodes"]],
        "arrows": [{"from": e["from"], "to": e["to"], "kind": e["kind"]} for e in data["edges"]],
        "problems": [_problem(p) for p in pipelines.check(data)],
    }


def _problem(p: dict) -> str:
    where = p["node"] or "pipeline"
    field = f" ({p['field']})" if p.get("field") else ""
    return f"{where}{field}: {p['message']}" + (" [warning]" if p["level"] == "warn" else "")


def start_turn(pipeline_id: str, message: str, token: str | None = None) -> dict:
    """Send a message: a turn starts working on it in its own process. ValueError when it can't start."""
    text = message.strip()[:MESSAGE_CHARS]
    if not text:
        raise ValueError("Write a message first.")
    if not settings()["model"]:
        raise ValueError("Choose the assistant's model in Settings first.")
    where = home(pipeline_id)
    if where.exists() and _turn(where).get("state") in WORKING:
        raise ValueError("The assistant is still working on your last message.")
    where.mkdir(exist_ok=True)
    chat = _chat(where)
    number = chat["turns"] + 1
    state = json.dumps(describe(pipelines.read(pipeline_id)), ensure_ascii=False)
    chat["messages"].append({"role": "user", "content": f"{text}\n\n<pipeline>\n{state}\n</pipeline>", "text": text,
                             "turn": number, "at": time.time()})
    chat["turns"] = number
    pipelines._write(where / "chat.json", chat)
    for name in ("stop.json", "go.json"):
        (where / name).unlink(missing_ok=True)
    pipelines._write(where / "turn.json", {"turn": number, "state": "starting", "doing": "Starting", "error": None,
                                            "waiting_for": None, "started": time.time(), "updated": time.time()})
    _spawn(where, number, token)
    return conversation(pipeline_id)


def _spawn(where: Path, number: int, token: str | None) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("ROCR_VISIBLE_DEVICES",)}
    env.update(HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(pipelines.SERVER_DIR),
               PYTHONUNBUFFERED="1", ANVIL_DATA=str(config.settings.data),
               ANVIL_DATASET_ROOTS=":".join(str(p) for p in config.settings.dataset_roots))
    if token:
        env["HF_TOKEN"] = token
    log = (where / "log.txt").open("ab")
    log.write(f"\n===== anvil.assistant turn {number} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n".encode())
    log.flush()
    _processes[str(where)] = subprocess.Popen(
        [sys.executable, "-m", "anvil.assistant", str(where), str(number)], cwd=pipelines.SERVER_DIR, env=env,
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log.close()
    turn = pipelines._read(where / "turn.json", {}) or {}
    turn["pid"] = _processes[str(where)].pid
    pipelines._write(where / "turn.json", turn)


def stop_turn(pipeline_id: str) -> None:
    """Stop the turn: it stops at once, and stops a preview it started."""
    where = home(pipeline_id)
    turn = _turn(where)
    if turn.get("state") not in WORKING:
        raise ValueError("The assistant isn't working on anything.")
    pipelines._write(where / "stop.json", {"requested": time.time()})
    if turn.get("pid") and _alive(where, turn["pid"]):
        try:
            os.kill(turn["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass


def ask_anyway(pipeline_id: str) -> None:
    """A turn waiting for its model's address to be free asks now."""
    where = home(pipeline_id)
    if _turn(where).get("state") != "waiting":
        raise ValueError("The assistant isn't waiting.")
    pipelines._write(where / "go.json", {"requested": time.time()})


def undo(pipeline_id: str) -> dict:
    """Put the pipeline back as it was before the newest turn that changed it."""
    where = home(pipeline_id)
    if _turn(where).get("state") in WORKING:
        raise ValueError("Wait for the assistant to finish, or stop it, first.")
    number = _undoable(where)
    if number is None:
        raise ValueError("There's nothing of the assistant's to undo.")
    before = pipelines._read(where / f"before-{number}.json")
    pipelines.save(pipeline_id, {"name": before["name"], "about": before.get("about", ""),
                                 "nodes": before["nodes"], "edges": before["edges"]})
    (where / f"before-{number}.json").unlink()
    chat = _chat(where)
    chat["messages"].append({"role": "note", "text": "Undid what the assistant changed after that message.",
                             "turn": number, "at": time.time()})
    pipelines._write(where / "chat.json", chat)
    return conversation(pipeline_id)


def clear(pipeline_id: str) -> None:
    """Start over: the conversation goes, the pipeline stays as it is."""
    where = home(pipeline_id)
    if where.exists() and _turn(where).get("state") in WORKING:
        raise ValueError("Wait for the assistant to finish, or stop it, first.")
    shutil.rmtree(where, ignore_errors=True)


# ---------------------------------------------------------------- what the model is told

STEP_GUIDE = {
    "model": """Starts a line; talks to an OpenAI-compatible endpoint. base_url: its address. model: one the endpoint
  serves (list_models). temperature 0-2. max_tokens. thinking: default, on or off (off for short answers and
  labels). parallel: requests at once (no more than the server answers at once). timeout in seconds. extra: more
  request fields, a JSON object. Its API key is pasted on the canvas: you never see or set it.""",
    "seed": """Rows from a dataset someone has: a file or a Hugging Face split. source: {"kind": "file", "path": ...}
  or {"kind": "hub", "repo": ..., "config": ..., "split": ...}. columns: the dataset's columns to bring in; later
  steps use them in lowercase with _ (Text Message becomes text_message). order: shuffled or in_order. limit: rows
  to use, 0 for all.""",
    "list": """A value for each row from a list. column: the new column's name. values: the list of strings. pick:
  random, or turns (each value in turn, for an even spread).""",
    "generate": """The model writes text from a prompt (Jinja, {{ column }} puts a column in). column: where the text
  goes. system: an optional system prompt. per_reply: 1 makes one row a request; more makes the model answer a
  JSON list, each item a row (faster and more varied). temperature: null uses the Model's.""",
    "question": """A decision question answered for each row. name: the answer's column. type: noul (yes or no),
  choice or score. instructions: the question. yes, no: what each answer means (noul). options: [{"name",
  "meaning"}] (choice). levels: what each level means, lowest first, 2 to 10 (score). about: the column the question
  is about. context: optional text the model also gets. answer: model (the model answers) or column (from_column's
  values through mapping, {value: answer or "leave out"}). sure_only: keep only answers the model is sure of.
  on_disagree: leave_out (the answer) or drop (the row), when a second Model on its check arrow disagrees.""",
    "filter": """Keeps the rows that pass its rules. rules: [{"column", "op", "value"}], op one of is, is_not,
  contains, not_contains, matches (a regular expression), longer, shorter (characters), empty, not_empty. match: all
  or any.""",
    "dedupe": """Drops exact and near-copies of a column. column. near: true also drops near-copies.""",
    "save": """Ends a line: writes its rows as a dataset. name: lowercase letters, digits, - _ and . rows: how many to
  make. format: decision (state_key, the state's field such as message, and state_column; the answers come from every
  Question above), chat (system, an optional Jinja prompt; user_column; assistant_column), pairs (prompt_column,
  completion_column) or text (text_column). balance: a List's column or a Question's name, for an equal share of
  rows for each of its values (optional).""",
}

SYSTEM = """You are anvil's pipeline assistant. anvil fine-tunes models, and its pipelines make the datasets they
train on: a model writes rows, answers questions about them, and the rows that fit are saved. You build and refine
one pipeline with the person you're talking to. They describe the data they want; you build the pipeline, preview
it, read the rows and improve it until it makes what they asked for.

How a pipeline works
- Steps joined by arrows. Every line of steps starts at a Model and ends at a Save. Rows flow down the rows arrows;
  each step adds a column to every row, or drops rows. A step can use, by name, the columns made above it on its
  line.
- A line can branch: each branch gets its own copy of every row, and each Save writes its own dataset.
- Generate and Question steps ask the Model their line starts from, unless another Model is plugged into them (a
  model arrow). A Question can also take a second opinion from a Model on its check arrow.
- Arrows are {"from", "to", "kind"}: kind rows, model or check. A step has one rows arrow coming in.

The steps, what each setting (config) means, and the defaults
STEPS_HERE

Rules
- Step ids: short, lowercase, unique, such as write, ask, save. Keep the Model's id (usually model) when you rebuild,
  so the API key pasted into it stays with it. You never see or set API keys: if a preview fails because the
  endpoint didn't accept the key, ask the person to paste it into the Model step.
- Columns, a Question's name and a List's column: lowercase letters, digits and _, starting with a letter, and
  unique on a line. A Question's name is a column too, so it can't be a List's column as well.
- Prompts are Jinja templates. Only columns made above the step exist.
- Lists that take turns or pick at random (topics, tones, situations, kinds of customer) make rows varied; so does
  per_reply over 1.
- A decision dataset (format decision) trains a model that answers questions about a text: its answers come from
  the Question steps above the Save, and balance gives each answer an equal share.
- A chat dataset: a Generate writes the user's message, a second Generate writes the reply from {{ that column }},
  and a Save with format chat takes the two columns.
- Use one model for every step, a second opinion included, unless the person asks for two: a server that holds one
  model at a time reloads on every switch, so two models there swap back and forth on almost every request.

How to work
1. Talk a new pipeline through before you build it. When someone describes the data they want, look at what helps
   (models, datasets, templates), ask what you still need to know (at most three short questions at a time), then
   propose the pipeline in a few plain lines: the steps in order, the model, the columns and answers, how many rows.
   Ask whether to build it, and wait for the answer.
2. Build when they say so (yes, go ahead, build it), or at once when they ask for a clear change to the pipeline.
   Build with set_pipeline (the whole pipeline at once) or add_step, change_step and remove_step (small changes).
   Each tells you what still stops the pipeline from running: fix that first.
3. Preview, then read the rows as a critic: realistic, varied, labelled right, in the right format? Fix the prompts
   or steps and preview again, at most three times in one go.
4. Finish with a short summary: what the pipeline makes, what you changed and why, what to look at. Full runs are
   the person's to start: say how many rows Save will make, and that Run (top right) makes them.
Questions about the pipeline, or about making data, are answered without changing anything.

Keep replies short: plain words, at most five short lines, one idea a line, no headings. The person can ask for
more, and the pipeline's notes can hold the detail.

Each message from the person ends with the pipeline as it is then, in <pipeline> tags, with the problems that stop
it from running. They may have changed it themselves since your last turn.

Templates you can study with show_template (working pipelines, with a note on every step):
TEMPLATES_HERE"""


def system_prompt() -> str:
    steps = "\n".join(f"{kind}: {STEP_GUIDE[kind]}\n  defaults: {json.dumps(pipelines.STEPS[kind])}"
                      for kind in pipelines.STEPS)
    shown = "\n".join(f"- {t['id']}: {t['name']}. {t['summary']}" for t in templates.public())
    return SYSTEM.replace("STEPS_HERE", steps).replace("TEMPLATES_HERE", shown)


def _tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": required or []}}}


STEP_TYPES = list(pipelines.STEPS)
TOOLS = [
    _tool("set_pipeline", "Replace the whole pipeline with these steps and arrows: to build one, or to restructure "
          "it. The steps are laid out on the canvas again.", {
              "name": {"type": "string", "description": "A short name for the pipeline"},
              "about": {"type": "string", "description": "A few sentences: what it makes and why it's built so"},
              "steps": {"type": "array", "items": {"type": "object", "properties": {
                  "id": {"type": "string"}, "type": {"type": "string", "enum": STEP_TYPES},
                  "note": {"type": "string", "description": "Why the step is there (optional)"},
                  "config": {"type": "object"}}, "required": ["id", "type", "config"]}},
              "arrows": {"type": "array", "items": {"type": "object", "properties": {
                  "from": {"type": "string"}, "to": {"type": "string"},
                  "kind": {"type": "string", "enum": ["rows", "model", "check"]}}, "required": ["from", "to"]}}},
          ["steps", "arrows"]),
    _tool("change_step", "Change some of one step's settings (the others stay as they are), its note, or both.", {
        "id": {"type": "string"}, "config": {"type": "object", "description": "Only the settings to change"},
        "note": {"type": "string"}}, ["id"]),
    _tool("add_step", "Add one step. A step goes after another (rows arrow): after one with steps below it, it goes in "
          "between, except a Save, which starts a branch. A Model can plug into a step (plug_into) or give a "
          "Question a second opinion (check).", {
              "type": {"type": "string", "enum": STEP_TYPES}, "id": {"type": "string"},
              "config": {"type": "object"}, "note": {"type": "string"},
              "after": {"type": "string", "description": "The step it goes after (any step but a Model)"},
              "plug_into": {"type": "string", "description": "For a Model: the Generate or Question it serves"},
              "check": {"type": "string", "description": "For a Model: the Question it gives a second opinion"}},
          ["type"]),
    _tool("remove_step", "Remove one step. The steps below it join the step above it.",
          {"id": {"type": "string"}}, ["id"]),
    _tool("preview", "Make a few rows (5 for each Save) and see what every step did: the rows kept and dropped, why, "
          "and what the model was asked and answered. The pipeline must have nothing left to fix."),
    _tool("list_models", "The models an endpoint serves: the pipeline's Model's address unless you give another.",
          {"base_url": {"type": "string"}}),
    _tool("list_datasets", "The dataset files a Seed data step can read (in anvil's dataset folders)."),
    _tool("look_at_dataset", "A dataset's rows, columns and format, with a few rows: a file (path) or a Hugging Face "
          "split (repo, config, split).", {
              "path": {"type": "string"}, "repo": {"type": "string"}, "config": {"type": "string"},
              "split": {"type": "string"}}),
    _tool("show_template", "One of anvil's templates in full (steps, settings, notes), to learn from or start "
          "from.", {"id": {"type": "string", "enum": [t["id"] for t in templates.TEMPLATES]}}, ["id"]),
]


# ---------------------------------------------------------------- the tools

class Context:
    """What the tools of one turn share."""

    def __init__(self, where: Path, number: int, token: str | None):
        self.where, self.number, self.token = where, number, token
        self.pipeline_id = where.parent.name
        self.previews = 0
        self.preview_started = False
        self.stop_requested = lambda: False
        self.doing = lambda text: None  # what the screen says the turn is doing

    def read(self) -> dict:
        return pipelines.read(self.pipeline_id)

    def save(self, data: dict, nodes: list[dict], edges: list[dict], name: str | None = None,
             about: str | None = None, tidy_up: bool = True) -> dict:
        """Save a changed graph, keeping what it was before this turn for Undo."""
        snapshot = self.where / f"before-{self.number}.json"
        if not snapshot.exists():
            pipelines._write(snapshot, data)
        graph = pipelines.normalize({"nodes": nodes, "edges": edges})
        if tidy_up:
            for node_id, (x, y) in tidy(graph).items():
                node = next(n for n in graph["nodes"] if n["id"] == node_id)
                node["x"], node["y"] = x, y
        _carry_keys(self.pipeline_id, data["nodes"], graph["nodes"])
        body = {**graph, "name": data["name"] if name is None else name}
        if about is not None:
            body["about"] = about
        return pipelines.save(self.pipeline_id, body)


def _carry_keys(pipeline_id: str, before: list[dict], after: list[dict]) -> None:
    """A Model that went away hands its API key to a Model at the same address that has none."""
    path = pipelines.folder(pipeline_id)
    keys = pipelines.read_keys(path)
    now = {n["id"] for n in after}
    moved = False
    for gone in (n for n in before if n["type"] == "model" and n["id"] in keys and n["id"] not in now):
        here = llm.base_url(str(gone["config"]["base_url"]))
        heir = next((n for n in after if n["type"] == "model" and n["id"] not in keys
                     and llm.base_url(str(n["config"]["base_url"])) == here), None)
        if heir:
            keys[heir["id"]] = keys.pop(gone["id"])
            moved = True
    if moved:
        pipelines._write(path / "secrets.json", keys, private=True)


def tidy(graph: dict) -> dict[str, tuple[float, float]]:
    """Where each step goes on the canvas: the canvas's Tidy (web/src/pipeline/graph.ts), so the two agree."""
    g = pipelines.Graph(graph)
    at: dict[str, tuple[float, float]] = {}
    column = [0]

    def place(node: str, depth: int, visiting: frozenset) -> float:
        kids = [k for k in g.children[node] if k not in visiting and k not in at]
        if not kids:
            x = column[0]
            column[0] += 1
        else:
            xs = [place(k, depth + 1, visiting | {node}) for k in kids]
            x = (xs[0] + xs[-1]) / 2
        at[node] = (x * (NODE_WIDTH + COLUMN_GAP), depth * ROW_GAP)
        return x

    plugs = set(g.plugged.values()) | set(g.second.values())
    for top in g.order:
        if top not in g.parent and not (g.nodes[top]["type"] == "model" and top in plugs and not g.children[top]) \
                and top not in at:
            place(top, 0, frozenset())
    for node in g.order:
        if node in at:
            continue
        served = next((s for s in g.order if g.plugged.get(s) == node or g.second.get(s) == node), None)
        if served in at:
            sx, sy = at[served]
            x = sx + NODE_WIDTH + 80
            while any(abs(py - sy) < 60 and abs(px - x) < NODE_WIDTH for px, py in at.values()):
                x += NODE_WIDTH + COLUMN_GAP
            at[node] = (x, sy)
        else:
            at[node] = (column[0] * (NODE_WIDTH + COLUMN_GAP), 0)
            column[0] += 1
    return at


def _coerce(kind: str, given: dict) -> dict:
    """A step's settings as the model wrote them, in the shapes they take: a list sent as JSON text or one item a
    line becomes a list, an object sent as JSON text an object, a number or true/false sent as text that."""
    found = {}
    for field, value in given.items():
        if field not in pipelines.STEPS[kind]:
            continue
        default = pipelines.STEPS[kind][field]
        if isinstance(value, str) and isinstance(default, (list, dict)):
            try:
                value = json.loads(value)
            except ValueError:
                value = [v.strip() for v in value.splitlines() if v.strip()] if isinstance(default, list) else {}
        elif isinstance(value, str) and isinstance(default, bool):
            value = value.strip().lower() in ("true", "yes", "1")
        elif isinstance(value, str) and isinstance(default, (int, float)) and not isinstance(default, bool):
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass
        found[field] = value
    return found


def _ignored(kind: str, given: dict) -> list[str]:
    return sorted(set(given) - set(pipelines.STEPS[kind]))


def _changed(saved: dict) -> dict:
    """What a change tool answers: what still stops the pipeline from running."""
    problems = [_problem(p) for p in pipelines.check(saved)]
    return {"ok": True, "problems": problems or "none: the pipeline can run"}


def _label(node: dict) -> str:
    return node["id"]


def use_set_pipeline(ctx: Context, args: dict) -> tuple[dict, str]:
    steps, arrows = args.get("steps"), args.get("arrows")
    if not isinstance(steps, list) or not isinstance(arrows, list):
        raise ToolError("steps and arrows are lists.")
    data = ctx.read()
    left_out, nodes, seen = [], [], set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        node_id, kind = str(step.get("id", "")), step.get("type")
        if kind not in pipelines.STEPS:
            left_out.append(f"{node_id or '?'}: {kind!r} isn't a step type ({', '.join(STEP_TYPES)})")
            continue
        if not pipelines.NODE_ID.match(node_id) or node_id in seen:
            left_out.append(f"{node_id!r}: ids are 1-64 letters, digits, - and _, each used once")
            continue
        seen.add(node_id)
        given = step.get("config") if isinstance(step.get("config"), dict) else {}
        ignored = _ignored(kind, given)
        if ignored:
            left_out.append(f"{node_id}: {kind} has no setting {', '.join(ignored)}")
        nodes.append({"id": node_id, "type": kind, "note": str(step.get("note") or ""), "config": _coerce(kind, given)})
    edges = [{"from": str(a.get("from", "")), "to": str(a.get("to", "")), "kind": a.get("kind") or "rows"}
             for a in arrows if isinstance(a, dict)]
    for edge in edges:
        if edge["from"] not in seen or edge["to"] not in seen:
            left_out.append(f"arrow {edge['from']} -> {edge['to']}: both ends must be steps")
    name = " ".join(str(args.get("name") or "").split())[:80] or None
    about = str(args["about"])[:pipelines.ABOUT_CHARS] if isinstance(args.get("about"), str) else None
    saved = ctx.save(data, nodes, edges, name=name, about=about)
    result = _changed(saved)
    if left_out:
        result["left_out"] = left_out
    errors = sum(1 for p in pipelines.check(saved) if p["level"] == "error")
    return result, f"Built the pipeline: {len(saved['nodes'])} steps" + (f", {errors} to fix" if errors else "")


def use_change_step(ctx: Context, args: dict) -> tuple[dict, str]:
    data = ctx.read()
    node = next((n for n in data["nodes"] if n["id"] == args.get("id")), None)
    if node is None:
        raise ToolError(f"No step is called {args.get('id')!r}. The steps: {', '.join(n['id'] for n in data['nodes'])}.")
    given = args.get("config") if isinstance(args.get("config"), dict) else {}
    nodes = copy.deepcopy(data["nodes"])
    target = next(n for n in nodes if n["id"] == node["id"])
    target["config"].update(_coerce(node["type"], given))
    shaped = pipelines.normalize({"nodes": [target], "edges": []})["nodes"][0]["config"]
    changed = [field for field in pipelines.STEPS[node["type"]] if shaped[field] != node["config"][field]]
    if isinstance(args.get("note"), str) and args["note"] != node.get("note", ""):
        target["note"] = args["note"]
        changed.append("note")
    if not changed:
        raise ToolError("That changes nothing: those are its settings already." if given else
                        f"Nothing to change: {node['type']} steps have {', '.join(pipelines.STEPS[node['type']])}.")
    saved = ctx.save(data, nodes, data["edges"], tidy_up=False)
    result = _changed(saved)
    ignored = _ignored(node["type"], given)
    if ignored:
        result["left_out"] = f"{node['type']} has no setting {', '.join(ignored)}"
    return result, f"Changed {_label(node)}: {', '.join(changed)}"


def _new_id(kind: str, taken: set[str]) -> str:
    if kind not in taken:
        return kind
    n = 2
    while f"{kind}-{n}" in taken:
        n += 1
    return f"{kind}-{n}"


def use_add_step(ctx: Context, args: dict) -> tuple[dict, str]:
    kind = args.get("type")
    if kind not in pipelines.STEPS:
        raise ToolError(f"{kind!r} isn't a step type ({', '.join(STEP_TYPES)}).")
    data = ctx.read()
    nodes, edges = copy.deepcopy(data["nodes"]), copy.deepcopy(data["edges"])
    ids = {n["id"] for n in nodes}
    node_id = str(args.get("id") or _new_id(kind, ids))
    if not pipelines.NODE_ID.match(node_id) or node_id in ids:
        raise ToolError(f"{node_id!r} is taken or isn't an id (1-64 letters, digits, - and _).")
    given = args.get("config") if isinstance(args.get("config"), dict) else {}
    nodes.append({"id": node_id, "type": kind, "note": str(args.get("note") or ""), "config": _coerce(kind, given)})
    by_id = {n["id"]: n for n in nodes}
    where = ""
    if kind == "model":
        for handle, field in (("model", "plug_into"), ("check", "check")):
            target = args.get(field)
            if target:
                if target not in by_id:
                    raise ToolError(f"No step is called {target!r}.")
                edges = [e for e in edges if not (e["to"] == target and e["kind"] == handle)]
                edges.append({"from": node_id, "to": target, "kind": handle})
                where = f" into {target}" if handle == "model" else f" as {target}'s second opinion"
    else:
        after = args.get("after")
        if after not in by_id or by_id[after]["type"] == "save":
            raise ToolError("Say which step it goes after (after): any step but a Save.")
        below = [e for e in edges if e["from"] == after and e["kind"] == "rows"]
        if kind != "save" and below:  # in between, as the canvas adds a step
            for edge in below:
                edge["from"] = node_id
        edges.append({"from": after, "to": node_id, "kind": "rows"})
        where = f" after {after}"
    saved = ctx.save(data, nodes, edges)
    result = _changed(saved)
    ignored = _ignored(kind, given)
    if ignored:
        result["left_out"] = f"{kind} has no setting {', '.join(ignored)}"
    result["id"] = node_id
    return result, f"Added {kind} step {node_id}{where}"


def use_remove_step(ctx: Context, args: dict) -> tuple[dict, str]:
    data = ctx.read()
    node_id = args.get("id")
    if node_id not in {n["id"] for n in data["nodes"]}:
        raise ToolError(f"No step is called {node_id!r}. The steps: {', '.join(n['id'] for n in data['nodes'])}.")
    above = next((e["from"] for e in data["edges"] if e["to"] == node_id and e["kind"] == "rows"), None)
    edges = []
    for edge in data["edges"]:
        if edge["to"] == node_id:
            continue
        if edge["from"] == node_id:
            if edge["kind"] == "rows" and above:  # the steps below join the step above
                edges.append({**edge, "from": above})
            continue
        edges.append(edge)
    saved = ctx.save(data, [n for n in data["nodes"] if n["id"] != node_id], edges)
    return _changed(saved), f"Removed {node_id}"


def _short(value, limit: int = TEXT_CHARS):
    """A value with its long texts cut, for a tool's result."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {k: _short(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_short(v, limit) for v in value[:20]]
    return value


def preview_summary(pipeline_id: str) -> dict:
    """What a preview made, for the assistant to read: each step's counts and drop reasons, the rows kept and some
    dropped, and what the model was asked and answered for the first rows."""
    status = pipelines.run_status(pipeline_id, "preview")
    file = pipelines.folder(pipeline_id) / "preview" / "preview.jsonl"
    entries = []
    if file.exists():
        for line in file.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    kept = [e for e in entries if e.get("outcome") == "kept"]
    dropped = [e for e in entries if e.get("outcome") != "kept"]
    asked = []
    for entry in kept[:2] + dropped[:1]:
        for step in entry.get("trace") or []:
            for ask in step.get("asked") or []:
                asked.append({"row": entry["id"], "step": step["node"], "prompt": _short(ask.get("prompt", ""),
                              PROMPT_CHARS), "reply": _short(ask.get("reply", ""), 600),
                              **({"answer": step["answer"]} if "answer" in step else {})})
    return {
        "state": status.get("state"), "error": status.get("error"),
        "saves": {k: {"kept": v.get("kept"), "target": v.get("target")} for k, v in (status.get("saves") or {}).items()},
        "steps": {k: {f: v.get(f, 0) for f in ("in", "out", "dropped", "errors", "cut")}
                  for k, v in (status.get("steps") or {}).items() if v.get("in") or v.get("dropped")},
        "drop_reasons": status.get("reasons") or {},
        "kept_rows": [{"row": _short(e.get("row")), "saved": _short({k: v for k, v in (e.get("record") or {}).items()
                                                                     if k not in ("meta", "source")})}
                      for e in kept[:5]],
        "dropped_rows": [{"step": e.get("node"), "reason": _short(e.get("reason", ""), 300), "row": _short(e.get("row"))}
                         for e in dropped[:6]],
        "asked": asked[:ASKED_SHOWN],
    }


def use_preview(ctx: Context, args: dict) -> tuple[dict, str] | tuple[dict, str, bool]:
    if ctx.previews >= MAX_PREVIEWS:
        raise ToolError(f"That's {MAX_PREVIEWS} previews in this turn: tell the person what you found, and let them "
                        "decide what's next.")
    deadline = time.monotonic() + PREVIEW_WAIT_S
    while pipelines.active(ctx.pipeline_id) == "preview":  # the person's own, started before this turn
        if ctx.stop_requested():
            raise Stopped
        if time.monotonic() > deadline:
            raise ToolError("A preview started on the canvas is still going after 15 minutes.")
        ctx.doing("Waiting for the preview that's going")
        time.sleep(1.5)
    try:
        pipelines.start(ctx.pipeline_id, True, ctx.token)
    except ValueError as error:
        raise ToolError(str(error)) from None
    ctx.previews += 1
    ctx.preview_started = True
    deadline = time.monotonic() + PREVIEW_WAIT_S
    while True:
        status = pipelines.run_status(ctx.pipeline_id, "preview")
        if status.get("state") not in pipelines.ACTIVE:
            break
        if ctx.stop_requested() or time.monotonic() > deadline:
            try:
                pipelines.stop(ctx.pipeline_id, "preview")
            except ValueError:
                pass
            if ctx.stop_requested():
                raise Stopped
            raise ToolError(f"The preview took longer than {PREVIEW_WAIT_S // 60} minutes, so it was stopped.")
        saves = status.get("saves") or {}
        made, target = sum(s.get("kept", 0) for s in saves.values()), sum(s.get("target", 0) for s in saves.values())
        ctx.doing(f"Previewing: {made} of {target} rows" if target else "Previewing")
        time.sleep(1.5)
    summary = preview_summary(ctx.pipeline_id)
    made = sum(s["kept"] or 0 for s in summary["saves"].values())
    target = sum(s["target"] or 0 for s in summary["saves"].values())
    if summary["state"] == "failed":
        return summary, f"The preview failed: {summary['error'] or 'see its log'}", False
    dropped = sum(s["dropped"] for s in summary["steps"].values())
    return summary, f"Previewed: {made} of {target} rows" + (f", {dropped} dropped" if dropped else "")


def use_list_models(ctx: Context, args: dict) -> tuple[dict, str]:
    data = ctx.read()
    model = next((n for n in data["nodes"] if n["type"] == "model"), None)
    url = str(args.get("base_url") or (model["config"]["base_url"] if model else settings()["base_url"]))
    key = pipelines.key_for(pipelines.folder(ctx.pipeline_id), data["nodes"], model["id"], url) if model \
        else key_at(url)
    try:
        base, found = llm.list_models(url, key)
    except llm.EndpointError as error:
        raise ToolError(str(error)) from None
    return {"base_url": base, "models": found}, f"Looked at the models at {base}"


def use_list_datasets(ctx: Context, args: dict) -> tuple[dict, str]:
    found = []
    for root in config.settings.all_dataset_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if len(found) >= 200:
                break
            if path.is_file() and path.suffix.lower() in (".parquet", ".jsonl", ".json", ".csv") \
                    and not any(part.startswith(".") for part in path.relative_to(root).parts):
                found.append({"path": str(path), "mb": round(path.stat().st_size / 1e6, 2)})
    return {"files": found}, f"Looked at the datasets ({len(found)} files)"


def use_look_at_dataset(ctx: Context, args: dict) -> tuple[dict, str]:
    from . import datasets  # pandas: only when a turn reads a dataset

    if args.get("path"):
        source = {"kind": "file", "path": str(args["path"])}
        try:
            folders.inside_roots(source["path"], config.settings.all_dataset_roots)
        except (PermissionError, FileNotFoundError) as error:
            raise ToolError(str(error)) from None
        label = Path(source["path"]).name
    elif args.get("repo"):
        source = {"kind": "hub", "repo": str(args["repo"]), "config": args.get("config") or None,
                  "split": str(args.get("split") or "train")}
        label = source["repo"]
    else:
        raise ToolError("Give a file's path, or a Hugging Face repo (and its config and split).")
    try:
        found = datasets.inspect(source, ctx.token)
    except Exception as error:  # noqa: BLE001 - whatever stops it being read is the answer
        raise ToolError(f"Couldn't read it: {error}") from None
    result = {"rows": found["rows"], "columns": found["columns"], "format": found["format"],
              "sample": _short(found["preview"]), "source": source}
    return result, f"Read {label}: {found['rows']:,} rows, columns {', '.join(found['columns'][:6])}"


def use_show_template(ctx: Context, args: dict) -> tuple[dict, str]:
    try:
        built = templates.build(str(args.get("id")))
    except FileNotFoundError:
        raise ToolError(f"No template is called {args.get('id')!r}.") from None
    graph = pipelines.normalize(built)
    return describe({"name": built["name"], "about": built["about"], **graph}), f"Read the template {built['name']}"


USES = {"set_pipeline": use_set_pipeline, "change_step": use_change_step, "add_step": use_add_step,
        "remove_step": use_remove_step, "preview": use_preview, "list_models": use_list_models,
        "list_datasets": use_list_datasets, "look_at_dataset": use_look_at_dataset,
        "show_template": use_show_template}
DOING = {"set_pipeline": "Building the pipeline", "change_step": "Changing a step", "add_step": "Adding a step",
         "remove_step": "Removing a step", "preview": "Previewing", "list_models": "Looking at the models",
         "list_datasets": "Looking at the datasets", "look_at_dataset": "Reading a dataset",
         "show_template": "Reading a template"}
FAILED = {"set_pipeline": "build the pipeline", "change_step": "change the step", "add_step": "add the step",
          "remove_step": "remove the step", "preview": "preview", "list_models": "list the models",
          "list_datasets": "list the datasets", "look_at_dataset": "read the dataset", "show_template": "read the template"}


class ToolError(Exception):
    """Why a tool couldn't do what it was asked, told to the model so it can try differently."""


def use(ctx: Context, call: llm.ToolCall) -> tuple[str, str, bool]:
    """Carry out one tool call: (its result as JSON, a line for the screen, whether it worked)."""
    if call.name not in USES:
        return json.dumps({"ok": False, "error": f"There's no tool called {call.name}."}), \
            f"Tried a tool that doesn't exist ({call.name})", False
    try:
        args = json.loads(call.arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError("not an object")
    except ValueError as error:
        return json.dumps({"ok": False, "error": f"The arguments weren't a JSON object: {error}"}), \
            f"Couldn't use {call.name}: its arguments weren't JSON", False
    try:
        result, summary, *rest = USES[call.name](ctx, args)  # a tool may say it didn't work: (result, line, False)
    except ToolError as error:
        return json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), \
            f"Couldn't {FAILED[call.name]}: {error}", False
    return json.dumps(result, ensure_ascii=False, default=str), summary, rest[0] if rest else True


# ---------------------------------------------------------------- a turn (python -m anvil.assistant)

def model_messages(entries: list[dict], budget: int, system: str) -> list[dict]:
    """The conversation as the model gets it, kept inside budget tokens: the pipeline as it was at older messages
    is left out first, then older tools' results (their one-line summaries stay), then the oldest turns."""
    convo: list[dict] = []
    for entry in entries:
        if entry["role"] == "user":
            convo.append({"role": "user", "content": entry["content"], "_turn": entry["turn"]})
        elif entry["role"] == "assistant":
            message = {"role": "assistant", "content": entry.get("content") or "", "_turn": entry["turn"]}
            if entry.get("tool_calls"):
                message["tool_calls"] = entry["tool_calls"]
            convo.append(message)
        elif entry["role"] == "tool":
            convo.append({"role": "tool", "tool_call_id": entry["tool_call_id"], "content": entry["content"],
                          "_turn": entry["turn"], "_summary": entry.get("summary", "")})
    newest = max((m["_turn"] for m in convo), default=0)

    def size() -> float:
        return (len(system) + sum(len(m["content"] or "") + len(json.dumps(m.get("tool_calls") or ""))
                                  for m in convo)) / CHARS_PER_TOKEN

    if size() > budget:
        for m in convo:
            if m["role"] == "user" and m["_turn"] < newest:
                m["content"] = re.sub(r"<pipeline>.*</pipeline>", "<pipeline>(as it was then: left out to save room)"
                                      "</pipeline>", m["content"], flags=re.S)
    if size() > budget:
        for m in convo:
            if m["role"] == "tool" and m["_turn"] < newest:
                m["content"] = json.dumps({"summary": m["_summary"], "result": "left out to save room"})
    while size() > budget and len({m["_turn"] for m in convo}) > 1:
        oldest = min(m["_turn"] for m in convo)
        convo = [m for m in convo if m["_turn"] != oldest]
    return [{"role": "system", "content": system},
            *({k: v for k, v in m.items() if not k.startswith("_")} for m in convo)]


def busy(s: dict) -> dict | None:
    """What the assistant has to wait for: training on the GPUs its server's models load onto (anvil.sharing), or a
    run (or preview) at its address with another model, since a server that holds one model at a time can't answer
    the assistant until that run lets go of it."""
    from . import sharing  # sharing builds on pipelines, as this module does

    mine = llm.base_url(s["base_url"])
    holder = sharing.held(mine)
    if holder:
        return {"training": holder, "base_url": mine}
    root = pipelines.pipelines_dir()
    for path in sorted(root.iterdir()) if root.exists() else []:
        try:
            going = pipelines.active(path.name)
        except FileNotFoundError:
            continue
        if not going:
            continue
        graph = pipelines._read((path / "preview" if going == "preview" else path / "runs" / going)
                                / "pipeline.json", {}) or {}
        for node in graph.get("nodes") or []:
            cfg = node.get("config") or {}
            if node.get("type") == "model" and llm.base_url(str(cfg.get("base_url", ""))) == mine \
                    and str(cfg.get("model", "")) != s["model"]:
                return {"pipeline": path.name, "name": graph.get("name", path.name), "run": going,
                        "model": str(cfg.get("model", "")), "base_url": mine}
    return None


class Turn:
    """turn.json, as this turn's process keeps it."""

    def __init__(self, where: Path, number: int):
        self.where, self.number = where, number

    def update(self, **fields) -> None:
        current = pipelines._read(self.where / "turn.json", {}) or {}
        if current.get("turn") != self.number:
            raise Stopped  # a newer turn or Start over replaced this one
        current.update(fields, updated=time.time())
        pipelines._write(self.where / "turn.json", current)

    def stop_requested(self) -> bool:
        return (self.where / "stop.json").exists()


def run(where: Path, number: int) -> None:
    """Work on the newest message until the model answers without calling a tool."""
    turn = Turn(where, number)
    ctx = Context(where, number, os.environ.get("HF_TOKEN") or None)
    ctx.stop_requested = turn.stop_requested
    ctx.doing = lambda text: turn.update(doing=text)
    s = settings()
    effort = {"reasoning_effort": s["reasoning_effort"]} if s["reasoning_effort"] != "default" else {}
    client = llm.Client(s["base_url"], key(), s["model"], temperature=s["temperature"], max_tokens=s["max_tokens"],
                        thinking=s["thinking"], extra=effort, timeout=s["timeout"], retries=2)
    system = system_prompt()
    chat = _chat(where)

    def add(entry: dict) -> None:
        chat["messages"].append({**entry, "turn": number, "at": time.time()})
        pipelines._write(where / "chat.json", chat)

    try:
        for _ in range(MAX_CALLS):
            if turn.stop_requested():
                raise Stopped
            asked_anyway = False
            while not asked_anyway and (waiting := busy(s)):
                turn.update(state="waiting", doing=None, waiting_for=waiting)
                for _ in range(WAIT_POLL_S * 2):
                    if turn.stop_requested():
                        raise Stopped
                    if (where / "go.json").exists():
                        asked_anyway = True
                        break
                    time.sleep(0.5)
            turn.update(state="working", doing="Thinking", waiting_for=None)
            budget = max(2000, s["context"] - s["max_tokens"] - 3000)  # 3000: the tools' descriptions
            answer = client.answer(model_messages(chat["messages"], budget, system), TOOLS)
            if not answer.calls and not answer.text:
                raise llm.EndpointError("The model answered nothing" + (
                    ": it hit max tokens, perhaps thinking. Raise Max tokens in Settings, or turn thinking off."
                    if answer.cut else "."))
            add({"role": "assistant", "content": answer.text, **({"tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in answer.calls]} if answer.calls else {})})
            if not answer.calls:
                turn.update(state="done", doing=None)
                return
            for call in answer.calls:
                if turn.stop_requested():
                    raise Stopped
                turn.update(doing=DOING.get(call.name, "Working"))
                result, summary, ok = use(ctx, call)
                add({"role": "tool", "tool_call_id": call.id, "tool": call.name, "content": result,
                     "summary": summary, "ok": ok})
        add({"role": "note", "text": f"The assistant stopped after {MAX_CALLS} steps without finishing. Ask it to "
                                     "carry on."})
        turn.update(state="done", doing=None)
    except Stopped:
        if ctx.preview_started:
            try:
                pipelines.stop(ctx.pipeline_id, "preview")
            except (ValueError, FileNotFoundError):
                pass
        if (where / "turn.json").exists() and (pipelines._read(where / "turn.json", {}) or {}).get("turn") == number:
            add({"role": "note", "text": "Stopped."})
            turn.update(state="stopped", doing=None, waiting_for=None)
    except llm.EndpointError as error:
        turn.update(state="failed", doing=None, waiting_for=None, error=f"The assistant's model: {error}")
    except Exception:  # noqa: BLE001 - say so on the screen; the log has the traceback
        traceback.print_exc()
        turn.update(state="failed", doing=None, waiting_for=None,
                    error="Something went wrong in anvil; the assistant's log says what.")
    finally:
        client.close()


def main() -> None:
    def stopped(*_):
        raise Stopped  # Stop: the turn stops wherever it is, a request to the model included

    signal.signal(signal.SIGTERM, stopped)
    run(Path(sys.argv[1]), int(sys.argv[2]))


if __name__ == "__main__":
    main()
