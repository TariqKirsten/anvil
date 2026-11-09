"""Pipelines: making datasets with a model. A pipeline is a graph of steps, drawn on the Pipelines page.

Every line of steps starts at a Model, which connects to an OpenAI-compatible endpoint (anvil.llm). Rows flow down the
solid arrows (rows edges), and each step adds a column to every row or drops some:

- seed      rows from a dataset you already have, a file or a Hugging Face split; its columns join the row
- list      a value from a list you type (a topic, a language, a tone), picked at random or taking turns
- generate  the model writes text from a prompt with {{ column }} slots: one row per reply, or several
- question  a decision question (yes/no, choice or score) answered for each row, by the model or from a column
- filter    keeps only the rows that pass its rules
- dedupe    drops copies and near-copies (70% of word trigrams shared) of a column
- save      ends a line: writes its rows as an anvil dataset (decision, chat, prompt and completion, or text),
            optionally with an equal share of rows for each value of a List's column or a Question's answer

A step that asks a model uses the Model its line starts from, unless another Model is plugged into it (a dashed model
edge). A question can also take a second opinion from a Model plugged into its check handle. Lines may branch: each
branch gets its own copy of every row, and each Save makes its own dataset.

On disk, in ANVIL_DATA/pipelines/<id>/:
- pipeline.json   the graph: {id, name, about, created, updated, nodes: [{id, type, x, y, note, config}], edges:
                  [{id, from, to, kind}]}, kind being rows, model or check; about and note are what people wrote about
                  the pipeline and its steps (a template's explanations), which the run ignores
- secrets.json    the Model steps' API keys, by node id; never sent back to the screen. A Model without a key of
                  its own uses the key of another Model at the same address, else the assistant's when it's at that
                  address (anvil.assistant), so a key is pasted once
- assistant/      the conversation with the pipeline assistant (anvil.assistant describes it)
- preview/        the newest preview (a few rows, every step's work shown)
- runs/<when>/    each full run (anvil.generate describes the files)

A full run's datasets go to ANVIL_DATA/datasets/generated/, where New run's data picker finds them.
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
from pathlib import Path

from jinja2 import Environment, TemplateSyntaxError, meta

from . import config, llm, templates

SERVER_DIR = Path(__file__).resolve().parents[1]
PIPELINE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
RUN_ID = re.compile(r"^\d{8}-\d{6}(-\d+)?$")
NODE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
COLUMN = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")
DATASET_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
ACTIVE = {"starting", "running", "stopping"}
EDGE_KINDS = {"rows", "model", "check"}
ASKS_MODEL = {"generate", "question"}  # steps a Model can plug into
MAX_ROWS = 1_000_000
NOTE_CHARS, ABOUT_CHARS = 2000, 6000

# Every step's settings, with their defaults. Settings a saved step lacks get these, so older pipelines keep working.
STEPS: dict[str, dict] = {
    "model": {"name": "", "base_url": "http://host.docker.internal:8000/v1", "model": "", "temperature": 0.8,
              "max_tokens": 2048, "thinking": "default", "parallel": 8, "timeout": 300, "extra": {}},
    "seed": {"source": None, "columns": [], "order": "shuffled", "limit": 0},
    "list": {"column": "", "values": [], "pick": "random"},
    "generate": {"column": "", "prompt": "", "system": "", "per_reply": 1, "temperature": None},
    "question": {"name": "", "type": "noul", "instructions": "", "yes": "", "no": "", "options": [], "levels": [],
                 "about": "", "context": "", "answer": "model", "from_column": "", "mapping": {},
                 "sure_only": False, "on_disagree": "leave_out"},
    "filter": {"rules": [], "match": "all"},
    "dedupe": {"column": "", "near": True},
    "save": {"name": "", "rows": 1000, "format": "decision", "state_key": "message", "state_column": "",
             "system": "", "user_column": "", "assistant_column": "", "prompt_column": "", "completion_column": "",
             "text_column": "", "balance": ""},
}
CHOICES = {
    ("model", "thinking"): ("default", "on", "off"),
    ("seed", "order"): ("shuffled", "in_order"),
    ("list", "pick"): ("random", "turns"),
    ("question", "type"): ("noul", "choice", "score"),
    ("question", "answer"): ("model", "column"),
    ("question", "on_disagree"): ("leave_out", "drop"),
    ("filter", "match"): ("all", "any"),
    ("save", "format"): ("decision", "chat", "pairs", "text"),
}
FILTER_OPS = {"is": "is", "is_not": "isn't", "contains": "contains", "not_contains": "doesn't contain",
              "matches": "matches", "longer": "is longer than", "shorter": "is shorter than", "empty": "is empty",
              "not_empty": "isn't empty"}
NEEDS_NO_VALUE = {"empty", "not_empty"}
LEAVE_OUT = "leave out"  # a mapping to this leaves a question unanswered for that value

_jinja = Environment()
_processes: dict[str, subprocess.Popen] = {}


# ---------------------------------------------------------------- the graph

def ident(name: str) -> str:
    """A column name a prompt can use as {{ name }}: a dataset's "Text Message" becomes text_message."""
    slug = re.sub(r"[^a-z0-9_]+", "_", str(name).lower()).strip("_")[:64]
    return slug if slug and not slug[0].isdigit() else f"c_{slug}"[:64] if slug else "column"


def values(config_values) -> list[str]:
    """A list step's values: one per line as typed, blank lines dropped."""
    return [str(v).strip() for v in config_values or [] if str(v).strip()]


def question_of(cfg: dict) -> dict:
    """A question step as a decision question, {type, instructions, criteria}, as datasets.check_decision takes it."""
    question: dict = {"type": cfg["type"], "instructions": cfg["instructions"].strip()}
    if cfg["type"] == "noul":
        criteria = {k: v for k, v in (("true", cfg["yes"].strip()), ("false", cfg["no"].strip())) if v}
        if criteria:
            question["criteria"] = criteria
    elif cfg["type"] == "choice":
        question["criteria"] = {name: str(o.get("meaning", "")).strip() for name, o in _options(cfg)}
    else:
        question["criteria"] = [str(level).strip() for level in cfg["levels"] if str(level).strip()]
    return question


def _options(cfg: dict) -> list[tuple[str, dict]]:
    """A choice question's options that have a name, as (name, option)."""
    return [(str(o.get("name", "")).strip(), o) for o in cfg["options"]
            if isinstance(o, dict) and str(o.get("name", "")).strip()]


def made(node: dict) -> list[str]:
    """The columns a step adds to each row."""
    cfg = node["config"]
    if node["type"] in ("list", "generate"):
        return [cfg["column"]] if cfg["column"] else []
    if node["type"] == "question":
        return [cfg["name"]] if cfg["name"] else []
    if node["type"] == "seed":
        return [ident(c) for c in cfg["columns"]]
    return []


def normalize(body: dict) -> dict:
    """A graph from the screen with every step's settings complete: unknown settings dropped, missing ones defaulted.
    It doesn't judge whether the graph can run (check does), so a pipeline half drawn can still be saved."""
    nodes, seen = [], set()
    for raw in body.get("nodes") or []:
        if not isinstance(raw, dict) or raw.get("type") not in STEPS:
            continue
        node_id = str(raw.get("id", ""))
        if not NODE_ID.match(node_id) or node_id in seen:
            continue
        seen.add(node_id)
        given = raw.get("config") if isinstance(raw.get("config"), dict) else {}
        cfg = copy.deepcopy(STEPS[raw["type"]])
        for key, default in STEPS[raw["type"]].items():
            if key in given:
                cfg[key] = _shaped(given[key], default)
        nodes.append({"id": node_id, "type": raw["type"], "x": _number(raw.get("x")), "y": _number(raw.get("y")),
                      "note": str(raw.get("note") or "")[:NOTE_CHARS], "config": cfg})
    edges, pairs = [], set()
    for raw in body.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        source, target, kind = str(raw.get("from", "")), str(raw.get("to", "")), raw.get("kind", "rows")
        if source in seen and target in seen and kind in EDGE_KINDS and source != target \
                and (source, target, kind) not in pairs:
            pairs.add((source, target, kind))
            edges.append({"id": str(raw.get("id") or f"{source}-{kind}-{target}")[:140], "from": source, "to": target,
                          "kind": kind})
    return {"nodes": nodes, "edges": edges}


def _shaped(value, default):
    """A setting in the shape its default has: text as text, lists and objects as such. A number stays whatever was
    sent, so check can say what's wrong with it (an emptied number field arrives as null)."""
    if isinstance(default, str):
        return "" if value is None else str(value)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, list):
        return value if isinstance(value, list) else []
    if isinstance(default, dict):
        return value if isinstance(value, dict) else {}
    if default is None:  # the seed's source, or a temperature left to the model
        return value if isinstance(value, (dict, int, float)) and not isinstance(value, bool) else None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(number, 1) if abs(number) < 1e7 else 0.0


class Graph:
    """The structure of a normalized graph: which step feeds which, the Model each step asks, and the problems with
    the wiring itself (check adds the ones in steps' settings)."""

    def __init__(self, graph: dict):
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.order = [n["id"] for n in graph["nodes"]]
        self.parent: dict[str, str] = {}
        self.children: dict[str, list[str]] = {n: [] for n in self.nodes}
        self.plugged: dict[str, str] = {}  # step -> the Model plugged into it
        self.second: dict[str, str] = {}  # question -> the Model giving a second opinion
        self.wiring: list[dict] = []
        for edge in graph["edges"]:
            source, target = self.nodes[edge["from"]], self.nodes[edge["to"]]
            if edge["kind"] == "rows":
                if source["type"] == "save":
                    self._problem(target["id"], "Save ends a line: nothing can come after it.")
                elif target["type"] == "model":
                    self._problem(target["id"], "A Model starts a line, so no rows flow into it. To use it in a "
                                                "step, plug it into the step's model handle.")
                elif target["id"] in self.parent:
                    self._problem(target["id"], "Rows can come into a step from one place only.")
                else:
                    self.parent[target["id"]] = source["id"]
                    self.children[source["id"]].append(target["id"])
                continue
            if source["type"] != "model":
                self._problem(target["id"], "Only a Model plugs into a model handle.")
            elif edge["kind"] == "model" and target["type"] not in ASKS_MODEL:
                self._problem(target["id"], "Only Generate and Question steps ask a model.")
            elif edge["kind"] == "check" and target["type"] != "question":
                self._problem(target["id"], "Only a Question step takes a second opinion.")
            else:
                plugs = self.plugged if edge["kind"] == "model" else self.second
                if target["id"] in plugs:
                    self._problem(target["id"], "One Model per handle: unplug the other one first.")
                else:
                    plugs[target["id"]] = source["id"]
        self.cyclic = {n for n in self.nodes if self._in_cycle(n)}

    def _problem(self, node: str | None, message: str, field: str | None = None, level: str = "error") -> None:
        self.wiring.append({"node": node, "field": field, "level": level, "message": message})

    def _in_cycle(self, node: str) -> bool:
        seen, here = set(), node
        while here in self.parent:
            if here in seen:
                return True
            seen.add(here)
            here = self.parent[here]
        return here in seen

    def path(self, node: str) -> list[str]:
        """The steps from the top of the line down to this one."""
        if node in self.cyclic:
            return [node]
        steps = [node]
        while steps[-1] in self.parent:
            steps.append(self.parent[steps[-1]])
        return steps[::-1]

    def root(self, node: str) -> str:
        return self.path(node)[0]

    def model_for(self, node: str) -> str | None:
        """The Model a step asks: the one plugged into it, or the one its line starts from."""
        if node in self.plugged:
            return self.plugged[node]
        top = self.root(node)
        return top if self.nodes[top]["type"] == "model" else None

    def below(self, node: str) -> list[str]:
        found, todo = [], list(self.children[node])
        while todo:
            here = todo.pop()
            if here not in found:
                found.append(here)
                todo += self.children[here]
        return found

    def saves_below(self, node: str) -> list[str]:
        return [n for n in [node, *self.below(node)] if self.nodes[n]["type"] == "save"]

    def above(self, node: str, kind: str) -> list[str]:
        """The steps of one type above this one on its line, top first."""
        return [n for n in self.path(node)[:-1] if self.nodes[n]["type"] == kind]

    def columns_above(self, node: str) -> list[str]:
        return [c for n in self.path(node)[:-1] for c in made(self.nodes[n])]

    def depth(self, node: str) -> int:
        return len(self.path(node)) - 1

    def starts(self) -> list[str]:
        """The Models that start a line."""
        return [n for n in self.order if self.nodes[n]["type"] == "model" and self.children[n]]


def balance_values(g: Graph, save: str) -> list[str] | None:
    """The values a Save balances its rows over, as text (true and false for a yes/no question), or None when it
    doesn't balance or the column isn't a List's or a Question's above it."""
    column = g.nodes[save]["config"]["balance"]
    if not column:
        return None
    for step in g.path(save)[:-1]:
        node = g.nodes[step]
        if node["type"] == "list" and node["config"]["column"] == column:
            return list(dict.fromkeys(values(node["config"]["values"])))
        if node["type"] == "question" and node["config"]["name"] == column:
            question = question_of(node["config"])
            if question["type"] == "noul":
                return ["true", "false"]
            if question["type"] == "choice":
                return list(question["criteria"])
            return [str(level) for level in range(len(question["criteria"]))]
    return None


def shares(total: int, keys: list[str]) -> dict[str, int]:
    """total rows split as evenly as they go: 5 over two values is 3 and 2."""
    return {key: total // len(keys) + (1 if i < total % len(keys) else 0) for i, key in enumerate(keys)} if keys else {}


def template_columns(text: str) -> set[str]:
    """The names a prompt template uses. TemplateSyntaxError if it isn't a valid template."""
    return meta.find_undeclared_variables(_jinja.parse(text))


def check(graph: dict) -> list[dict]:
    """Everything that stops a graph from running (level error) or that it should know (level warn), each with the
    step it's about (node; None for the pipeline as a whole) and the setting (field) when there is one."""
    g = Graph(graph)
    problems = list(g.wiring)

    def add(node, message, field=None, level="error"):
        problems.append({"node": node, "field": field, "level": level, "message": message})

    if not g.nodes:
        add(None, "Add a Model to start the pipeline.")
        return problems
    for node_id in g.cyclic:
        add(node_id, "These steps go round in a circle. Rows need a line from a Model down to a Save.")
    saves = [n for n in g.order if g.nodes[n]["type"] == "save"]
    if not saves:
        add(None, "End a line with a Save step: it writes the rows as a dataset.")
    names: dict[str, str] = {}
    for node_id in g.order:
        node = g.nodes[node_id]
        kind, cfg = node["type"], node["config"]
        if kind != "model" and node_id not in g.parent and node_id not in g.cyclic:
            add(node_id, "Connect this step: draw an arrow into it from the step before. Every line starts at a Model.")
        if node_id in g.cyclic:
            continue
        have = g.columns_above(node_id)
        unused = kind == "model" and not g.children[node_id] and node_id not in g.plugged.values() \
            and node_id not in g.second.values()
        if unused:  # a Model only half set up doesn't stop the rest from running
            add(node_id, "This Model isn't used yet: draw an arrow from it, or plug it into a step.", level="warn")
        else:
            problems += _check_step(g, node_id, have)
        for column in made(node):
            if column in have:
                field = "columns" if kind == "seed" else "name" if kind == "question" else "column"
                add(node_id, f"There's already a column called {column} above this step.", field)
        if kind == "save" and cfg["name"]:
            if cfg["name"] in names:
                add(node_id, "Another Save writes a dataset with this name.", "name")
            names[cfg["name"]] = node_id
        if kind != "save" and kind != "model" and not g.saves_below(node_id):
            add(node_id, "Nothing below this step saves its rows: end the line with Save.", level="warn")
    return problems


def _check_step(g: Graph, node_id: str, have: list[str]) -> list[dict]:
    node = g.nodes[node_id]
    kind, cfg = node["type"], node["config"]
    found: list[dict] = []

    def add(message, field=None, level="error"):
        found.append({"node": node_id, "field": field, "level": level, "message": message})

    def column(field, what="Pick a column", optional=False):
        value = cfg[field]
        if not value:
            if not optional:
                add(f"{what}.", field)
        elif value not in have:
            add(f"There's no column called {value} above this step{_listing(have)}.", field)

    def name(field, what):
        if not cfg[field]:
            add(f"Name the {what}.", field)
        elif not COLUMN.match(cfg[field]):
            add(f"Use lowercase letters, digits and _ for the {what}, starting with a letter: such as "
                f"{ident(cfg[field])}.", field)

    def template(field, required):
        text = cfg[field]
        if not isinstance(text, str) or not text.strip():
            if required:
                add("Write the prompt.", field)
            return
        try:
            used = template_columns(text)
        except TemplateSyntaxError as error:
            add(f"The template doesn't work: {error.message} (line {error.lineno}).", field)
            return
        missing = sorted(used - set(have))
        if missing:
            add(f"{{{{ {missing[0]} }}}} isn't a column above this step{_listing(have)}.", field)

    def number(field, low, high, what, integer=True, optional=False):
        value = cfg[field]
        if value is None and optional:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (integer and value != int(value)) \
                or not low <= value <= high:
            add(f"{what} goes from {low:,} to {high:,}." if integer else f"{what} goes from {low} to {high}.", field)

    for (step, field), allowed in CHOICES.items():
        if step == kind and cfg[field] not in allowed:
            add(f"Pick one of: {', '.join(allowed)}.", field)

    if kind == "model":
        problem = llm.check_url(llm.base_url(str(cfg["base_url"])))
        if problem:
            add(problem, "base_url")
        if not str(cfg["model"]).strip():
            add("Pick the model to use: Connect lists the ones the endpoint serves.", "model")
        number("temperature", 0, 2, "Temperature", integer=False, optional=True)
        number("max_tokens", 1, 1_000_000, "Max tokens")
        number("parallel", 1, 256, "Requests at once")
        number("timeout", 5, 3600, "The timeout (seconds)")
        if not isinstance(cfg["extra"], dict):
            add("Extra request fields must be a JSON object.", "extra")
        elif {"model", "messages"} & set(cfg["extra"]):
            add("Extra request fields can't set model or messages: the step does.", "extra")
    elif kind == "seed":
        source = cfg["source"]
        if not isinstance(source, dict) or source.get("kind") not in ("file", "hub"):
            add("Pick the dataset to start from.", "source")
        if not cfg["columns"]:
            add("Bring in at least one column.", "columns")
        elif len({ident(c) for c in cfg["columns"]}) < len(cfg["columns"]):
            add("Two columns get the same name here. Leave one of them out.", "columns")
        number("limit", 0, 100_000_000, "Rows to use (0 for all)")
    elif kind == "list":
        name("column", "column")
        if not values(cfg["values"]):
            add("Add at least one value, one per line.", "values")
    elif kind == "generate":
        name("column", "column")
        template("prompt", True)
        template("system", False)
        number("per_reply", 1, 100, "Rows per reply")
        number("temperature", 0, 2, "Temperature", integer=False, optional=True)
    elif kind == "question":
        name("name", "question")
        if not str(cfg["instructions"]).strip():
            add("Write the question.", "instructions")
        question = question_of(cfg)
        if cfg["type"] == "choice":
            names = [name for name, _ in _options(cfg)]
            if len(names) < 2:
                add("Give at least two options.", "options")
            elif len(set(names)) < len(names):
                add("Two options have the same name.", "options")
            elif len(names) > 255:
                add("A choice can have at most 255 options.", "options")
        if cfg["type"] == "score" and not 2 <= len(question["criteria"]) <= 10:
            add("Describe 2 to 10 levels, lowest first.", "levels")
        asks = cfg["answer"] == "model" or node_id in g.second
        column("about", "Pick the column the question is about", optional=not asks)
        if cfg["answer"] == "column":
            column("from_column", "Pick the column the answers come from")
            for value, answer in (cfg["mapping"] or {}).items():
                if answer != LEAVE_OUT and _answer(question, answer) is None:
                    add(f"{answer!r} isn't an answer this question can have (for {value!r}).", "mapping")
                    break
    elif kind == "filter":
        if not cfg["rules"]:
            add("Add at least one rule.", "rules")
        for rule in cfg["rules"]:
            if not isinstance(rule, dict) or rule.get("op") not in FILTER_OPS:
                add("Every rule needs a column, a test and a value.", "rules")
                break
            if not rule.get("column"):
                add("Pick the column each rule tests.", "rules")
                break
            if rule["column"] not in have:
                add(f"There's no column called {rule['column']} above this step{_listing(have)}.", "rules")
                break
            value = str(rule.get("value", ""))
            if rule["op"] in ("longer", "shorter") and not value.strip().isdigit():
                add("Give a length as a whole number of characters.", "rules")
                break
            if rule["op"] == "matches":
                try:
                    re.compile(value)
                except re.error as error:
                    add(f"That pattern doesn't work: {error}.", "rules")
                    break
    elif kind == "dedupe":
        column("column", "Pick the column to compare")
    elif kind == "save":
        if not cfg["name"]:
            add("Name the dataset.", "name")
        elif not DATASET_NAME.match(cfg["name"]):
            add("Use lowercase letters, digits, - _ and . for the dataset's name, such as bank-messages.", "name")
        number("rows", 1, MAX_ROWS, "Rows")
        if cfg["balance"]:
            if cfg["balance"] not in have:
                add(f"There's no column called {cfg['balance']} above this step{_listing(have)}.", "balance")
            elif balance_values(g, node_id) is None:
                add("Balance by a List's column or a Question's answer: those are the columns whose values are "
                    "known before the rows are made.", "balance")
        fmt = cfg["format"]
        if fmt == "decision":
            if not str(cfg["state_key"]).strip():
                add("Name the state's field, such as message.", "state_key")
            column("state_column", "Pick the column that goes in the state")
            if not g.above(node_id, "question"):
                add("A decision dataset needs a Question step above its Save: that's where its answers come from.")
        elif fmt == "chat":
            template("system", False)
            column("user_column", "Pick the column for the user's turn")
            column("assistant_column", "Pick the column for the assistant's reply")
        elif fmt == "pairs":
            column("prompt_column", "Pick the column for the prompt")
            column("completion_column", "Pick the column for the completion")
        elif fmt == "text":
            column("text_column", "Pick the column for the text")
    if kind in ASKS_MODEL and g.model_for(node_id) is None:
        add("This step needs a Model: start its line at one, or plug one in.")
    return found


def _listing(have: list[str]) -> str:
    return f" (there's {', '.join(dict.fromkeys(have))})" if have else " (there's none yet)"


def _answer(question: dict, value):
    """A value as this question's answer: True or False, an option's name, or a level number. None if it can't be."""
    if isinstance(value, str):
        value = value.strip()
    if question["type"] == "noul":
        if isinstance(value, bool):
            return value
        text = str(value).lower()
        return True if text in ("true", "yes", "1") else False if text in ("false", "no", "0") else None
    if question["type"] == "choice":
        options = question["criteria"]
        if value in options:
            return value
        lowered = {name.lower(): name for name in options}
        return lowered.get(str(value).lower())
    if isinstance(value, bool):
        return None
    try:
        level = int(value)
    except (TypeError, ValueError):
        return None
    return level if 0 <= level < len(question["criteria"]) and str(level) == str(value).strip() else None


def answer(question: dict, value, mapping: dict | None = None):
    """A column's value as the answer, through the step's mapping when it names the value."""
    if mapping and str(value) in mapping:
        value = mapping[str(value)]
        if value == LEAVE_OUT:
            return None
    return _answer(question, value)


# ---------------------------------------------------------------- storing pipelines

def pipelines_dir() -> Path:
    return config.settings.data / "pipelines"


def _read(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _write(path: Path, data, private: bool = False) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    if private:
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def folder(pipeline_id: str) -> Path:
    if not PIPELINE_ID.match(pipeline_id):
        raise FileNotFoundError("No such pipeline.")
    path = pipelines_dir() / pipeline_id
    if not (path / "pipeline.json").exists():
        raise FileNotFoundError("No such pipeline.")
    return path


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "pipeline"


def create(name: str, template: str | None = None) -> dict:
    """A new pipeline: just a Model, where every line starts, or a copy of a template (anvil.templates)."""
    start = templates.build(template) if template else {
        "name": "New pipeline", "about": "", "nodes": [{"id": "model", "type": "model", "x": 0, "y": 0, "config": {}}],
        "edges": []}
    name = " ".join(name.split())[:80] or start["name"]
    base = _slug(name)
    pipeline_id, n = base, 2
    while (pipelines_dir() / pipeline_id).exists():
        pipeline_id, n = f"{base}-{n}", n + 1
    path = pipelines_dir() / pipeline_id
    path.mkdir(parents=True)
    now = time.time()
    data = {"id": pipeline_id, "name": name, "about": start["about"][:ABOUT_CHARS], "created": now, "updated": now,
            **normalize(start)}
    _write(path / "pipeline.json", data)
    return data


def read(pipeline_id: str) -> dict:
    data = _read(folder(pipeline_id) / "pipeline.json")
    if not data:
        raise FileNotFoundError("No such pipeline.")
    return {"about": "", **data, **normalize(data)}


def save(pipeline_id: str, body: dict) -> dict:
    """Save the graph (and the name) as the screen has it now."""
    path = folder(pipeline_id)
    data = read(pipeline_id)
    name = " ".join(str(body.get("name", data["name"])).split())[:80]
    data.update(normalize(body), name=name or data["name"], updated=time.time())
    if "about" in body:
        data["about"] = str(body["about"] or "")[:ABOUT_CHARS]
    _write(path / "pipeline.json", data)
    return data


def delete(pipeline_id: str) -> None:
    path = folder(pipeline_id)
    if active(pipeline_id):
        raise ValueError("This pipeline is running. Stop it first.")
    shutil.rmtree(path)


def list_pipelines() -> list[dict]:
    found = []
    root = pipelines_dir()
    if root.exists():
        for path in root.iterdir():
            data = _read(path / "pipeline.json")
            if not data:
                continue
            try:
                latest = next(iter(runs(path.name)), None)
            except FileNotFoundError:  # deleted while the list was being made
                continue
            found.append({"id": data["id"], "name": data["name"], "updated": data.get("updated"),
                          "steps": len(data.get("nodes") or []), "latest": latest})
    return sorted(found, key=lambda p: p.get("updated") or 0, reverse=True)


def detail(pipeline_id: str) -> dict:
    from . import assistant  # anvil.assistant builds on this module

    data = read(pipeline_id)
    keys = read_keys(folder(pipeline_id))
    preview = run_status(pipeline_id, "preview") if (folder(pipeline_id) / "preview" / "status.json").exists() else None
    s = assistant.settings()
    return {**data, "keys": sorted(k for k in keys if k in {n["id"] for n in data["nodes"]}),
            "problems": check(data), "runs": runs(pipeline_id), "preview": preview,
            "assistant_key_at": s["base_url"] if s["has_key"] else None}


# ---------------------------------------------------------------- API keys

def read_keys(path: Path) -> dict[str, str]:
    return _read(path / "secrets.json", {}) or {}


def key_for(path: Path, nodes: list[dict], node: str, url: str | None = None) -> str | None:
    """A Model step's key: its own, else one saved for another Model step at the same address (url: the address the
    screen has now, when it isn't saved yet), else the assistant's when the assistant is at that address."""
    from . import assistant  # anvil.assistant builds on this module

    keys = read_keys(path)
    if node in keys:
        return keys[node]
    here = llm.base_url(url if url is not None else next(
        (str(n["config"]["base_url"]) for n in nodes if n["id"] == node), ""))
    return next((keys[n["id"]] for n in nodes if n["type"] == "model" and n["id"] in keys
                 and llm.base_url(str(n["config"]["base_url"])) == here), None) or assistant.key_at(here)


def set_key(pipeline_id: str, node: str, key: str | None) -> None:
    """Keep a Model step's API key on the server (None removes it)."""
    path = folder(pipeline_id)
    if not NODE_ID.match(node):
        raise FileNotFoundError("No such step.")
    keys = read_keys(path)
    if key:
        keys[node] = key
    else:
        keys.pop(node, None)
    _write(path / "secrets.json", keys, private=True)


def _model_config(pipeline_id: str, node: str, given: dict | None) -> dict:
    """A Model step's settings as the screen has them now (it may not have saved them yet), else as saved."""
    cfg = copy.deepcopy(STEPS["model"])
    if given:
        cfg.update({k: v for k, v in given.items() if k in cfg})
    else:
        saved = next((n for n in read(pipeline_id)["nodes"] if n["id"] == node and n["type"] == "model"), None)
        if saved is None:
            raise FileNotFoundError("No such Model step.")
        cfg.update(saved["config"])
    return cfg


def endpoint_models(pipeline_id: str, node: str, url: str) -> dict:
    """The models an endpoint serves, asked with the step's saved key (or one for the same address)."""
    key = key_for(folder(pipeline_id), read(pipeline_id)["nodes"], node, url)
    base, models = llm.list_models(url, key)
    return {"base_url": base, "models": models}


def try_model(pipeline_id: str, node: str, given: dict | None, message: str) -> dict:
    """One message to a Model step, as it's set up now: its reply and how long it took."""
    cfg = _model_config(pipeline_id, node, given)
    problem = llm.check_url(llm.base_url(str(cfg["base_url"])))
    if problem:
        raise llm.EndpointError(problem, fatal=True)
    if not str(cfg["model"]).strip():
        raise llm.EndpointError("Pick the model first.", fatal=True)
    key = key_for(folder(pipeline_id), read(pipeline_id)["nodes"], node, str(cfg["base_url"]))
    client = llm.Client(cfg["base_url"], key, str(cfg["model"]),
                        temperature=cfg["temperature"], max_tokens=cfg["max_tokens"], thinking=cfg["thinking"],
                        extra=cfg["extra"] if isinstance(cfg["extra"], dict) else {},
                        timeout=min(float(cfg["timeout"]), 180), retries=0)
    try:
        reply = client.chat([{"role": "user", "content": message}])
    finally:
        client.close()
    return {"reply": reply.text, "seconds": round(reply.seconds, 2), "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens, "cut": reply.cut}


# ---------------------------------------------------------------- runs (anvil.generate)

def _run_folder(pipeline_id: str, run_id: str) -> Path:
    path = folder(pipeline_id)
    if run_id == "preview":
        return path / "preview"
    if not RUN_ID.match(run_id):
        raise FileNotFoundError("No such run.")
    return path / "runs" / run_id


def _alive(run: Path, pid: int | None) -> bool:
    proc = _processes.get(str(run))
    if proc is not None:
        return proc.poll() is None  # poll() also reaps the finished process
    if not pid:
        return False
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return b"anvil.generate" in args and str(run).encode() in args


def _status(run: Path) -> dict:
    """A run's status, corrected when its process ended without saying so (a crash, or the container restarting)."""
    current = _read(run / "status.json", {}) or {}
    if current.get("state") in ACTIVE and not _alive(run, current.get("pid")):
        if (run / "stop.json").exists():
            current.update(state="stopped", stage="Stopped")
        else:
            current.update(state="failed", stage="Failed", error=current.get("error") or (
                "The run ended without finishing. Its log may say why; Resume carries on from the rows it kept."))
        _write(run / "status.json", current)
    return current


def run_status(pipeline_id: str, run_id: str) -> dict:
    run = _run_folder(pipeline_id, run_id)
    if not (run / "status.json").exists():
        raise FileNotFoundError("No such run.")
    return {"id": run_id, **{k: v for k, v in _status(run).items() if k != "pid"}}


def runs(pipeline_id: str) -> list[dict]:
    """Every full run of a pipeline, newest first."""
    path = folder(pipeline_id) / "runs"
    if not path.exists():
        return []
    return [run_status(pipeline_id, run.name) for run in sorted(path.iterdir(), reverse=True)
            if RUN_ID.match(run.name) and (run / "status.json").exists()]


def active(pipeline_id: str) -> str | None:
    """The run (or "preview") of this pipeline that's still going."""
    path = folder(pipeline_id)
    candidates = [path / "preview"] + (sorted((path / "runs").iterdir(), reverse=True) if (path / "runs").exists()
                                       else [])
    for run in candidates:
        if (run / "status.json").exists() and _status(run).get("state") in ACTIVE:
            return "preview" if run.name == "preview" else run.name
    return None


def _spawn(run: Path, pipeline: Path, token: str | None) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("ROCR_VISIBLE_DEVICES",)}
    env.update(HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(SERVER_DIR), PYTHONUNBUFFERED="1",
               ANVIL_DATA=str(config.settings.data),  # the runner reads and writes where this server does
               ANVIL_DATASET_ROOTS=":".join(str(p) for p in config.settings.dataset_roots))
    if token:
        env["HF_TOKEN"] = token
    (run / "stop.json").unlink(missing_ok=True)
    log = (run / "log.txt").open("ab")
    log.write(f"\n===== anvil.generate {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n".encode())
    log.flush()
    _processes[str(run)] = subprocess.Popen(
        [sys.executable, "-m", "anvil.generate", str(pipeline), str(run)], cwd=SERVER_DIR, env=env, stdout=log,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log.close()
    current = _read(run / "status.json", {}) or {}
    current.update(state="starting", stage="Starting", error=None, pid=_processes[str(run)].pid,
                   updated=time.time())
    _write(run / "status.json", current)


def start(pipeline_id: str, preview: bool, token: str | None) -> str:
    """Start a preview (a few rows, replacing the last preview) or a full run. Returns the run's id."""
    run_id, run = new_run(pipeline_id, preview)
    _spawn(run, folder(pipeline_id), token)
    return run_id


def new_run(pipeline_id: str, preview: bool) -> tuple[str, Path]:
    """A new run's id and folder, ready for anvil.generate: the pipeline as it is now, and whether it's a preview.
    ValueError when the pipeline can't run yet, or is running already."""
    from . import sharing  # sharing builds on this module

    path = folder(pipeline_id)
    data = read(pipeline_id)
    errors = [p for p in check(data) if p["level"] == "error"]
    if errors:
        raise ValueError(errors[0]["message"] if len(errors) == 1 else
                         f"{errors[0]['message']} ({len(errors) - 1} more to fix: see the steps marked in red.)")
    going = active(pipeline_id)
    if going:
        raise ValueError("A preview of this pipeline is still going." if going == "preview" else
                         "This pipeline is running. Stop it first, or wait for it to finish.")
    for node in data["nodes"]:
        holder = sharing.held(str(node["config"]["base_url"])) if node["type"] == "model" else None
        if holder:
            raise ValueError(f"{holder} is training on the GPUs the models at {llm.base_url(node['config']['base_url'])}"
                             " load onto. Run this when training ends.")
    if preview:
        run_id, run = "preview", path / "preview"
        shutil.rmtree(run, ignore_errors=True)
    else:
        base = time.strftime("%Y%m%d-%H%M%S")
        run_id, n = base, 1
        while (path / "runs" / run_id).exists():
            run_id, n = f"{base}-{n}", n + 1
        run = path / "runs" / run_id
    run.mkdir(parents=True)
    _write(run / "pipeline.json", {k: data[k] for k in ("id", "name", "nodes", "edges")})
    _write(run / "request.json", {"preview": preview, "requested": time.time()})
    _write(run / "status.json", {"state": "starting", "stage": "Starting", "preview": preview,
                                 "started": time.time(), "updated": time.time()})
    return run_id, run


def stop(pipeline_id: str, run_id: str) -> None:
    """Stop feeding new rows: the requests under way get up to 30 s, then the rows kept so far are saved."""
    run = _run_folder(pipeline_id, run_id)
    current = _status(run)
    if current.get("state") not in ACTIVE:
        raise ValueError("This run isn't going.")
    _write(run / "stop.json", {"requested": time.time()})
    pid = current.get("pid")
    if _alive(run, pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def resume(pipeline_id: str, run_id: str, token: str | None) -> None:
    """Carry on a full run that was stopped or failed, from the rows it kept."""
    run = _run_folder(pipeline_id, run_id)
    if run_id == "preview":
        raise ValueError("A preview can't be resumed: start a new one.")
    current = _status(run)
    if current.get("state") not in ("stopped", "failed"):
        raise ValueError("Only a run that stopped or failed can carry on.")
    if active(pipeline_id):
        raise ValueError("Something else of this pipeline is running. Wait for it, or stop it.")
    _spawn(run, folder(pipeline_id), token)


def delete_run(pipeline_id: str, run_id: str) -> None:
    """Delete a run's folder. The datasets it wrote stay where they are, in case a training run uses them."""
    run = _run_folder(pipeline_id, run_id)
    if not run.exists():
        raise FileNotFoundError("No such run.")
    if _status(run).get("state") in ACTIVE:
        raise ValueError("This run is still going. Stop it first.")
    shutil.rmtree(run)
    _processes.pop(str(run), None)


def log(pipeline_id: str, run_id: str, lines: int = 200) -> str:
    """The end of a run's log: what its process printed, tracebacks included."""
    file = _run_folder(pipeline_id, run_id) / "log.txt"
    if not file.exists():
        raise FileNotFoundError("No such run.")
    return "\n".join(file.read_text(errors="replace").splitlines()[-lines:])


def rows(pipeline_id: str, run_id: str, which: str, offset: int = 0, limit: int = 50) -> dict:
    """Rows of a run to look at: a Save's kept rows (which = its node id), or "dropped" (with the step and reason).
    A preview's rows are all in preview.jsonl, kept and dropped together, each with what every step did."""
    run = _run_folder(pipeline_id, run_id)
    if which == "preview":
        file = run / "preview.jsonl"
    elif which == "dropped":
        file = run / "dropped.jsonl"
    elif NODE_ID.match(which):
        file = run / "rows" / f"{which}.jsonl"
    else:
        raise FileNotFoundError("No such rows.")
    found, total = [], 0
    if file.exists():
        with file.open(encoding="utf-8") as f:
            for line in f:
                if not line.endswith("\n"):
                    break  # still being written
                if offset <= total < offset + limit:
                    found.append(json.loads(line))
                total += 1
    return {"rows": found, "total": total, "offset": offset}
