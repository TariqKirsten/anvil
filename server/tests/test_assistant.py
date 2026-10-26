"""The pipeline assistant: its settings and key, the tools it uses on a pipeline, a turn from message to answer (in
this process and in its own), waiting while a run holds its address, Stop, Undo, and keeping inside its context."""

import json
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from anvil import assistant, llm, pipelines
from test_pipelines import arrow, client, fake, labelled, node  # noqa: F401 - fake is a fixture


class Scripted:
    """An OpenAI-compatible endpoint that answers with the replies it's given, one a request: {"content": text} or
    {"tool_calls": [(name, arguments), ...]}."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        scripted = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, data):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send(200, {"object": "list", "data": [{"id": "helper"}, {"id": "other"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with scripted.lock:
                    scripted.requests.append(body)
                    reply = scripted.replies.pop(0) if scripted.replies else {"content": "(nothing more scripted)"}
                calls = [{"id": f"call_{len(scripted.requests)}_{i}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args)}}
                         for i, (name, args) in enumerate(reply.get("tool_calls", []))]
                message = {"role": "assistant", "content": reply.get("content")}
                if calls:
                    message["tool_calls"] = calls
                self.send(200, {"choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}],
                                "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def say(text):
    return {"content": text}


def calls(*pairs):
    return {"tool_calls": list(pairs)}


def as_tool(graph):
    """A graph from test_pipelines as set_pipeline takes it."""
    return {"name": "Frustrated customers",
            "steps": [{"id": n["id"], "type": n["type"], "config": n["config"]} for n in graph["nodes"]],
            "arrows": [{"from": e["from"], "to": e["to"], "kind": e["kind"]} for e in graph["edges"]]}


@pytest.fixture
def scripted():
    endpoints = []

    def make(*replies):
        endpoint = Scripted(replies)
        endpoints.append(endpoint)
        assistant.save_settings({"base_url": endpoint.url, "model": "helper"})
        return endpoint

    yield make
    for endpoint in endpoints:
        endpoint.close()


def begin(pipeline_id, message, monkeypatch):
    """A turn started as the server starts one, but left for the test to run here (where it has no process of its
    own, so it counts as alive)."""
    monkeypatch.setattr(assistant, "_spawn", lambda where, number, token: None)
    monkeypatch.setattr(assistant, "_alive", lambda where, pid: True)
    assistant.start_turn(pipeline_id, message)
    return assistant.home(pipeline_id), assistant._chat(assistant.home(pipeline_id))["turns"]


def wait_for(check, seconds=30):
    deadline = time.time() + seconds
    while not check():
        assert time.time() < deadline
        time.sleep(0.1)


def test_settings_keep_the_key_on_the_server(settings):
    c = client()
    shown = c.get("/api/assistant").json()["settings"]
    assert shown["model"] == "" and shown["has_key"] is False
    saved = c.put("/api/assistant", json={"base_url": "http://host:9000/v1/chat/completions", "model": " helper ",
                                          "context": 65536}).json()["settings"]
    assert (saved["base_url"], saved["model"], saved["context"]) == ("http://host:9000/v1", "helper", 65536)
    assert c.put("/api/assistant", json={"temperature": 3}).status_code == 400
    assert c.put("/api/assistant", json={"reasoning_effort": "extreme"}).status_code == 400
    assert c.put("/api/assistant", json={"reasoning_effort": "xhigh"}).json()["settings"]["reasoning_effort"] == "xhigh"
    assert c.put("/api/assistant", json={"base_url": "host:9000"}).status_code == 400
    assert c.put("/api/assistant/key", json={"key": "sk-secret"}).json()["settings"]["has_key"] is True
    secrets = settings.data / "assistant" / "secrets.json"
    assert stat.S_IMODE(secrets.stat().st_mode) == 0o600
    assert "sk-secret" not in c.get("/api/assistant").text
    assert c.delete("/api/assistant/key").json()["settings"]["has_key"] is False


def test_a_turn_builds_the_pipeline_previews_it_and_answers(settings, fake, scripted, monkeypatch):
    pid = pipelines.create("Bank")["id"]
    endpoint = scripted(calls(("set_pipeline", as_tool(labelled(fake.url)))), calls(("preview", {})),
                        say("It makes frustrated and calm bank messages."))
    where, number = begin(pid, "Make messages from frustrated and calm bank customers", monkeypatch)
    assistant.run(where, number)

    data = pipelines.read(pid)
    assert [n["id"] for n in data["nodes"]] == ["m", "tone", "write", "ask", "dedupe", "save"]
    assert data["name"] == "Frustrated customers"
    assert len({(n["x"], n["y"]) for n in data["nodes"]}) == 6  # laid out, not piled up
    shown = assistant.conversation(pid)
    assert [m["role"] for m in shown["messages"]] == ["user", "action", "action", "assistant"]
    assert shown["messages"][1]["text"] == "Built the pipeline: 6 steps"
    assert shown["messages"][2]["text"].startswith("Previewed: 5 of 5 rows")
    assert shown["turn"]["state"] == "done" and shown["undo"] == 1

    first, second, third = endpoint.requests
    assert first["model"] == "helper" and len(first["tools"]) == len(assistant.TOOLS)
    assert "reasoning_effort" not in first  # default: left to the server
    assert first["messages"][0]["role"] == "system" and "<pipeline>" in first["messages"][-1]["content"]
    assert second["messages"][-1]["role"] == "tool"
    assert json.loads(second["messages"][-1]["content"])["problems"] == "none: the pipeline can run"
    preview = json.loads(third["messages"][-1]["content"])
    assert preview["saves"]["save"] == {"kept": 5, "target": 5} and len(preview["kept_rows"]) == 5
    assert preview["asked"] and "message" in preview["kept_rows"][0]["row"]

    assistant.undo(pid)
    assert [n["id"] for n in pipelines.read(pid)["nodes"]] == ["model"]
    assert assistant.conversation(pid)["messages"][-1]["role"] == "note"
    with pytest.raises(ValueError):
        assistant.undo(pid)


def test_steps_are_added_changed_and_removed(settings):
    pid = pipelines.create("Bank")["id"]
    pipelines.save(pid, labelled("http://127.0.0.1:1/v1"))
    where = assistant.home(pid)
    where.mkdir()
    ctx = assistant.Context(where, 1, None)

    result, summary = assistant.use_add_step(ctx, {"type": "filter", "after": "write", "config": {
        "rules": [{"column": "message", "op": "longer", "value": "10"}]}})
    assert result["id"] == "filter" and summary == "Added filter step filter after write"
    g = pipelines.Graph(pipelines.read(pid))
    assert g.parent["filter"] == "write" and g.parent["ask"] == "filter"  # in between
    assert (where / "before-1.json").exists()

    result, summary = assistant.use_change_step(ctx, {"id": "write", "config": {"per_reply": "5", "nope": 1}})
    assert summary == "Changed write: per_reply" and "nope" in result["left_out"]
    assert next(n for n in pipelines.read(pid)["nodes"] if n["id"] == "write")["config"]["per_reply"] == 5
    resent = {**next(n for n in pipelines.read(pid)["nodes"] if n["id"] == "write")["config"],
              "prompt": "Write {{ tone }} messages about loans"}  # the whole config, one setting new
    assert assistant.use_change_step(ctx, {"id": "write", "config": resent})[1] == "Changed write: prompt"
    with pytest.raises(assistant.ToolError, match="changes nothing"):
        assistant.use_change_step(ctx, {"id": "write", "config": {"per_reply": 5}})

    assistant.use_remove_step(ctx, {"id": "filter"})
    assert pipelines.Graph(pipelines.read(pid)).parent["ask"] == "write"

    assistant.use_add_step(ctx, {"type": "save", "after": "ask", "id": "all", "config": {
        "name": "all-rows", "format": "text", "text_column": "message"}})
    assistant.use_add_step(ctx, {"type": "model", "id": "judge", "check": "ask", "config": {
        "base_url": "http://127.0.0.1:1/v1", "model": "fake-model"}})
    g = pipelines.Graph(pipelines.read(pid))
    assert sorted(g.children["ask"]) == ["all", "dedupe"] and g.second["ask"] == "judge"
    assert [p for p in pipelines.check(pipelines.read(pid)) if p["level"] == "error"] == []

    with pytest.raises(assistant.ToolError):
        assistant.use_change_step(ctx, {"id": "missing", "config": {}})
    with pytest.raises(assistant.ToolError):
        assistant.use_add_step(ctx, {"type": "filter"})  # after which step?
    out, line, ok = assistant.use(ctx, llm.ToolCall("1", "remove_step", "{not json"))
    assert not ok and json.loads(out)["ok"] is False


def test_a_rebuilt_model_keeps_its_key(settings):
    pid = pipelines.create("Bank")["id"]
    pipelines.save(pid, labelled("http://127.0.0.1:1/v1"))
    pipelines.set_key(pid, "m", "sk-model")
    where = assistant.home(pid)
    where.mkdir()
    graph = labelled("http://127.0.0.1:1/v1")
    graph["nodes"][0]["id"] = "gen"
    graph["edges"][0]["from"] = "gen"
    assistant.use_set_pipeline(assistant.Context(where, 1, None), as_tool(graph))
    assert pipelines.read_keys(pipelines.folder(pid)) == {"gen": "sk-model"}


def test_a_model_without_a_key_uses_the_assistants_at_its_address(settings):
    assistant.save_settings({"base_url": "http://host:8000/v1", "model": "helper"})
    assistant.set_key("sk-assistant")
    pid = pipelines.create("Bank")["id"]
    nodes = [node("m", "model", base_url="http://host:8000/v1/"), node("far", "model", base_url="http://far:1/v1")]
    path = pipelines.folder(pid)
    assert pipelines.key_for(path, nodes, "m") == "sk-assistant"
    assert pipelines.key_for(path, nodes, "far") is None


def test_a_turn_waits_while_a_run_holds_its_address(settings, scripted, monkeypatch):
    endpoint = scripted(say("Hello"))
    busy = pipelines.create("Busy")["id"]
    run = pipelines.folder(busy) / "runs" / "20261024-120000"
    run.mkdir(parents=True)
    pipelines._write(run / "pipeline.json", {"name": "Busy", "nodes": [
        node("m", "model", base_url=endpoint.url, model="other")], "edges": []})
    real = pipelines.active
    monkeypatch.setattr(pipelines, "active", lambda pid: "20261024-120000" if pid == busy else real(pid))
    pid = pipelines.create("Mine")["id"]

    where, number = begin(pid, "hi", monkeypatch)
    worker = threading.Thread(target=assistant.run, args=(where, number))
    worker.start()
    wait_for(lambda: assistant.conversation(pid)["turn"]["state"] == "waiting")
    waiting = assistant.conversation(pid)["turn"]["waiting_for"]
    assert (waiting["pipeline"], waiting["model"]) == (busy, "other") and not endpoint.requests
    assistant.ask_anyway(pid)
    worker.join(30)
    assert assistant.conversation(pid)["turn"]["state"] == "done" and len(endpoint.requests) == 1

    where, number = begin(pid, "and again", monkeypatch)  # Stop while it waits
    worker = threading.Thread(target=assistant.run, args=(where, number))
    worker.start()
    wait_for(lambda: assistant.conversation(pid)["turn"]["state"] == "waiting")
    assistant.stop_turn(pid)
    worker.join(30)
    shown = assistant.conversation(pid)
    assert shown["turn"]["state"] == "stopped" and shown["messages"][-1] == {
        "role": "note", "text": "Stopped.", "at": shown["messages"][-1]["at"]}


def test_the_server_runs_a_turn_in_its_own_process(settings, scripted, monkeypatch):
    c = client()
    endpoint = scripted(say("Tell me what the data is for."))
    assistant.save_settings({"reasoning_effort": "medium"})
    pid = c.post("/api/pipelines", json={"name": "Asked"}).json()["pipeline"]["id"]
    started = c.post(f"/api/pipelines/{pid}/assistant", json={"message": "Make me some data"}).json()
    assert started["messages"][0] == {"role": "user", "text": "Make me some data", "at": started["messages"][0]["at"]}
    deadline = time.time() + 60
    while (shown := c.get(f"/api/pipelines/{pid}/assistant").json())["turn"]["state"] in assistant.WORKING:
        assert time.time() < deadline
        time.sleep(0.25)
    assert shown["turn"]["state"] == "done", shown
    assert shown["messages"][-1]["text"] == "Tell me what the data is for."
    assert endpoint.requests[0]["tools"] == assistant.TOOLS  # a conversation, free form: every tool, every message
    assert endpoint.requests[0]["reasoning_effort"] == "medium"
    assert c.post(f"/api/pipelines/{pid}/assistant", json={"message": " "}).status_code == 409

    monkeypatch.setattr(assistant, "working", lambda pipeline_id: True)
    assert c.put(f"/api/pipelines/{pid}", json={"name": "Asked", "nodes": [], "edges": []}).status_code == 409
    assert c.post(f"/api/pipelines/{pid}/runs", json={"preview": True}).status_code == 409
    assert c.delete(f"/api/pipelines/{pid}").status_code == 409


def test_the_conversation_stays_inside_the_models_context():
    big = "x" * 3000
    entries = []
    for turn in (1, 2, 3):
        entries += [
            {"role": "user", "content": f"message {turn}\n\n<pipeline>\n{big}\n</pipeline>", "turn": turn},
            {"role": "assistant", "content": "", "turn": turn, "tool_calls": [
                {"id": f"c{turn}", "type": "function", "function": {"name": "preview", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{turn}", "content": big, "summary": f"Previewed {turn}", "turn": turn},
            {"role": "assistant", "content": f"answer {turn}", "turn": turn},
        ]
    whole = assistant.model_messages(entries, 100_000, "system")
    assert len(whole) == 13 and big in whole[1]["content"]
    trimmed = assistant.model_messages(entries, 3000, "system")
    users = [m["content"] for m in trimmed if m["role"] == "user"]
    assert big in users[-1] and all(big not in u for u in users[:-1])
    tools = [m["content"] for m in trimmed if m["role"] == "tool"]
    assert tools[-1] == big and "Previewed 1" in tools[0]
    smallest = assistant.model_messages(entries, 1500, "system")
    assert [m["content"] for m in smallest if m["role"] == "user"][0].startswith("message 3")
    for i, message in enumerate(smallest):  # every call keeps its result
        for call in message.get("tool_calls") or []:
            assert smallest[i + 1]["tool_call_id"] == call["id"]


def test_steps_are_laid_out_as_the_canvas_lays_them_out():
    graph = pipelines.normalize({"nodes": [node("m", "model"), node("a", "generate"), node("b", "save"),
                                           node("c", "save"), node("judge", "model")],
                                 "edges": [arrow("m", "a"), arrow("a", "b"), arrow("a", "c"),
                                           arrow("judge", "a", "model")]})
    at = assistant.tidy(graph)
    step = assistant.NODE_WIDTH + assistant.COLUMN_GAP
    assert at["b"] == (0, 2 * assistant.ROW_GAP) and at["c"] == (step, 2 * assistant.ROW_GAP)
    assert at["a"] == (step / 2, assistant.ROW_GAP) and at["m"] == (step / 2, 0)
    assert at["judge"][1] == assistant.ROW_GAP and at["judge"][0] > at["a"][0]  # beside the step it serves


def test_tool_calls_come_back_from_the_endpoint(scripted, settings):
    endpoint = scripted(calls(("preview", {}), ("change_step", {"id": "write", "config": {"per_reply": 3}})))
    answer = llm.Client(endpoint.url, None, "helper").answer([{"role": "user", "content": "go"}], assistant.TOOLS)
    assert [c.name for c in answer.calls] == ["preview", "change_step"]
    assert json.loads(answer.calls[1].arguments) == {"id": "write", "config": {"per_reply": 3}}
    assert endpoint.requests[0]["tools"] == assistant.TOOLS
