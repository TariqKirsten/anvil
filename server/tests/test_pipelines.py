"""Pipelines: the graph's rules, keys that stay on the server, the endpoint calls, and runs that make datasets."""

import json
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from anvil import datasets, generate, pipelines
from anvil.main import app


class Fake:
    """An OpenAI-compatible endpoint. Asked for a list, it writes numbered messages from the prompt's first line; asked
    a question, it says yes when the text says angry (or whatever judge says)."""

    def __init__(self, key=None, delay=0.0, judge=None):
        self.key, self.delay, self.judge = key, delay, judge
        self.requests: list[dict] = []
        self.count = 0
        self.lock = threading.Lock()
        fake = self

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

            def allowed(self):
                if fake.key and self.headers.get("Authorization") != f"Bearer {fake.key}":
                    self.send(401, {"error": {"message": "Invalid API key"}})
                    return False
                return True

            def do_GET(self):
                if self.path != "/v1/models":
                    self.send(404, {"error": {"message": "Not found"}})
                elif self.allowed():
                    self.send(200, {"object": "list", "data": [{"id": "fake-model"}, {"id": "another"}]})

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send(404, {"error": {"message": "Not found"}})
                    return
                if not self.allowed():
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.requests.append(body)
                time.sleep(fake.delay)
                self.send(200, {"choices": [{"message": {"role": "assistant", "content": fake.reply(body)},
                                             "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 12, "completion_tokens": 7}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, body):
        prompt = body["messages"][-1]["content"]
        if "Answer with a JSON object" in prompt:
            text = prompt.split("<<<\n", 1)[1].split("\n>>>", 1)[0]
            return json.dumps({"answer": self.judge(text) if self.judge else "angry" in text, "sure": True})
        many = re.search(r"JSON array of (\d+) strings", prompt)
        if many:
            n = int(many.group(1))
            with self.lock:
                first, self.count = self.count, self.count + n
            return json.dumps([f"{prompt.splitlines()[0]} {k} {k * 7919 % 997} {k * 104729 % 991}"
                               for k in range(first, first + n)])
        return f"You said: {prompt}"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake():
    endpoint = Fake()
    yield endpoint
    endpoint.close()


def client():
    c = TestClient(app)
    assert c.post("/api/login", json={"username": "admin", "password": "correct horse"}).status_code == 200
    return c


def node(node_id, kind, **cfg):
    return {"id": node_id, "type": kind, "x": 0, "y": 0, "config": cfg}


def arrow(source, target, kind="rows"):
    return {"from": source, "to": target, "kind": kind}


def labelled(url, wanted=6, parallel=4):
    """Model → List (tone, taking turns) → Generate (three a reply) → Question (frustrated?) → Dedupe → Save."""
    return {"nodes": [
        node("m", "model", base_url=url, model="fake-model", parallel=parallel),
        node("tone", "list", column="tone", values=["an angry", "a calm", ""], pick="turns"),
        node("write", "generate", column="message", prompt="Write {{ tone }} message about cards", per_reply=3),
        node("ask", "question", name="frustrated", type="noul", instructions="Is the customer frustrated?",
             about="message"),
        node("dedupe", "dedupe", column="message"),
        node("save", "save", name="bank", rows=wanted, format="decision", state_key="message",
             state_column="message"),
    ], "edges": [arrow("m", "tone"), arrow("tone", "write"), arrow("write", "ask"), arrow("ask", "dedupe"),
                 arrow("dedupe", "save")]}


def errors(graph):
    return [p for p in pipelines.check(pipelines.normalize(graph)) if p["level"] == "error"]


def run_now(pipeline_id, preview=False, one_at_a_time=False):
    """Run the pipeline here rather than in a process of its own. one_at_a_time: each row from the top runs to its
    end before the next starts, so which rows a Save keeps doesn't depend on which request answers first."""
    run_id, run = pipelines.new_run(pipeline_id, preview)
    runner = generate.Run(pipelines.folder(pipeline_id), run)
    if one_at_a_time:
        runner.workers = runner.most_waiting = 1
    return run_id, run, runner.run()


def kept_rows(run, save="save"):
    return [json.loads(line) for line in (run / "rows" / f"{save}.jsonl").read_text().splitlines()]


def status(run):
    return json.loads((run / "status.json").read_text())


# ---------------------------------------------------------------- the graph

def test_a_new_pipeline_is_just_a_model(settings):
    made = pipelines.create("Bank messages")
    assert made["id"] == "bank-messages" and [n["type"] for n in made["nodes"]] == ["model"]
    assert pipelines.create("Bank messages")["id"] == "bank-messages-2"
    problems = pipelines.check(pipelines.read("bank-messages"))
    assert [p["message"] for p in problems if p["level"] == "error"] == [
        "End a line with a Save step: it writes the rows as a dataset."]
    assert [p["node"] for p in problems if p["level"] == "warn"] == ["model"]  # not used yet, so not checked either


def test_every_template_is_ready_to_run(settings):
    from anvil import templates

    shown = templates.public()
    assert [t["id"] for t in shown] == ["label-messages", "yes-no-from-scratch", "route-to-team", "several-questions",
                                         "chat-replies", "harmless-af-zu"]
    ids = {}
    for t in shown:
        made = pipelines.create("", t["id"])
        ids[t["id"]] = made["id"]
        assert made["name"] == t["name"] and made["about"]
        assert all(n["note"] for n in made["nodes"]), t["id"]  # every step says why it's there
        assert [p for p in pipelines.check(made) if p["level"] == "error"] == [], t["id"]
        assert {n["config"]["base_url"] for n in made["nodes"] if n["type"] == "model"} == {
            "http://host.docker.internal:8000/v1"}
    assert ids["label-messages"] == "label-messages-you-already-have"
    seed = next(n for n in pipelines.read(ids["label-messages"])["nodes"] if n["type"] == "seed")
    assert seed["config"]["source"]["path"] == str(settings.uploads / "anvil-sample-bank-messages.jsonl")
    assert len(datasets.read_file(settings.uploads / "anvil-sample-bank-messages.jsonl")) == 24


def test_notes_and_about_survive_saving(settings):
    pid = pipelines.create("Notes")["id"]
    graph = pipelines.read(pid)
    graph["nodes"][0]["note"] = "x" * 3000
    pipelines.save(pid, {**graph, "about": "What this is for."})
    kept = pipelines.read(pid)
    assert kept["about"] == "What this is for." and kept["nodes"][0]["note"] == "x" * pipelines.NOTE_CHARS
    pipelines.save(pid, {"nodes": kept["nodes"], "edges": []})  # a save without about leaves it alone
    assert pipelines.read(pid)["about"] == "What this is for."


def test_a_template_runs(settings, fake):
    made = pipelines.create("Labelled", "label-messages")
    graph = pipelines.read(made["id"])
    for n in graph["nodes"]:
        if n["type"] == "model":
            n["config"].update(base_url=fake.url, model="fake-model")
    pipelines.save(made["id"], graph)
    _, run, code = run_now(made["id"])
    assert code == 0 and status(run)["saves"]["save"]["kept"] == 24  # the sample's 24, then its seed ran out
    texts = [json.loads(r["state"])["message"] for r in kept_rows(run)]
    assert len(set(texts)) == 24


def test_a_complete_graph_has_nothing_to_fix(settings):
    assert errors(labelled("http://host.docker.internal:8000/v1")) == []


def test_check_says_what_stops_a_graph(settings):
    graph = {"nodes": [
        node("m", "model", base_url="localhost:8000", model=""),
        node("write", "generate", column="message", prompt="About {{ topik }}"),
        node("loose", "list", column="Topic!", values=[" "]),
        node("save", "save", name="out", format="decision", state_column="message"),
        node("after", "list", column="x", values=["1"]),
    ], "edges": [arrow("m", "write"), arrow("write", "save"), arrow("loose", "write", "model"),
                 arrow("save", "after"), arrow("loose", "m")]}
    found = {(p["node"], p["field"], p["message"]) for p in errors(graph)}
    assert ("m", "base_url", "An endpoint's address starts with http:// or https://, such as "
                             "http://host.docker.internal:8000/v1.") in found
    assert ("m", "model", "Pick the model to use: Connect lists the ones the endpoint serves.") in found
    assert ("write", "prompt", "{{ topik }} isn't a column above this step (there's none yet).") in found
    assert ("write", None, "Only a Model plugs into a model handle.") in found
    assert ("loose", "column", "Use lowercase letters, digits and _ for the column, starting with a letter: such as "
                               "topic.") in found
    assert ("loose", "values", "Add at least one value, one per line.") in found
    assert ("loose", None, "Connect this step: draw an arrow into it from the step before. Every line starts at a "
                           "Model.") in found
    assert ("m", None, "A Model starts a line, so no rows flow into it. To use it in a step, plug it into the step's "
                       "model handle.") in found
    assert ("after", None, "Save ends a line: nothing can come after it.") in found
    assert ("save", None, "A decision dataset needs a Question step above its Save: that's where its answers come "
                          "from.") in found


def test_columns_come_from_above_and_never_twice(settings):
    graph = labelled("http://x:1/v1")
    graph["nodes"][2]["config"]["column"] = "tone"  # Generate writes into the list's column
    graph["nodes"][3]["config"]["about"] = "nothing"
    graph["nodes"].append(node("two", "list", column="b", values=["1"]))
    graph["edges"] += [arrow("m", "two"), arrow("two", "write")]  # a second way into Generate
    messages = {p["message"] for p in errors(graph)}
    assert "There's already a column called tone above this step." in messages
    assert "There's no column called nothing above this step (there's tone)." in messages
    assert "Rows can come into a step from one place only." in messages


def test_steps_going_round_in_a_circle_are_refused(settings):
    graph = {"nodes": [node("a", "list", column="a", values=["1"]), node("b", "list", column="b", values=["1"])],
             "edges": [arrow("a", "b"), arrow("b", "a")]}
    assert {p["node"] for p in errors(graph) if "circle" in p["message"]} == {"a", "b"}


def test_settings_keep_their_shape(settings):
    graph = pipelines.normalize({"nodes": [
        {"id": "q", "type": "question", "config": {"options": "nope", "sure_only": 0, "name": 5, "api_key": "sk-x",
                                                   "unknown": 1, "levels": None}},
        {"id": "bad id!", "type": "list"}, {"id": "z", "type": "nonsense"}, {"id": "q", "type": "list"},
    ], "edges": [{"from": "q", "to": "missing"}, {"from": "q", "to": "q"}]})
    assert [n["id"] for n in graph["nodes"]] == ["q"] and graph["edges"] == []
    cfg = graph["nodes"][0]["config"]
    assert cfg["options"] == [] and cfg["levels"] == [] and cfg["sure_only"] is False and cfg["name"] == "5"
    assert "api_key" not in cfg and "unknown" not in cfg and cfg["type"] == "noul"


def test_question_answers_from_columns_follow_the_question(settings):
    choice = pipelines.question_of({**pipelines.STEPS["question"], "type": "choice", "instructions": "Which?",
                                    "options": [{"name": "Cards", "meaning": "card trouble"}, {"name": "loans"}]})
    assert choice == {"type": "choice", "instructions": "Which?", "criteria": {"Cards": "card trouble", "loans": ""}}
    assert pipelines.answer(choice, "cards") == "Cards" and pipelines.answer(choice, "fees") is None
    assert pipelines.answer(choice, "x", {"x": "loans"}) == "loans"
    assert pipelines.answer(choice, "x", {"x": pipelines.LEAVE_OUT}) is None
    yes_no = pipelines.question_of({**pipelines.STEPS["question"], "instructions": "Angry?", "yes": "shouting"})
    assert yes_no["criteria"] == {"true": "shouting"}
    assert [pipelines.answer(yes_no, v) for v in (True, "yes", "No", "0", "maybe")] == [True, True, False, False, None]
    score = pipelines.question_of({**pipelines.STEPS["question"], "type": "score", "instructions": "How bad?",
                                   "levels": ["mild", "bad", " "]})
    assert score["criteria"] == ["mild", "bad"]
    assert [pipelines.answer(score, v) for v in (1, "0", 2, "1.5", True)] == [1, 0, None, None, None]


# ---------------------------------------------------------------- keys and endpoints

def test_keys_stay_on_the_server(settings):
    fake = Fake(key="sk-right")
    try:
        c = client()
        pid = c.post("/api/pipelines", json={"name": "Keys"}).json()["pipeline"]["id"]
        assert c.put(f"/api/pipelines/{pid}/keys/model", json={"key": " sk-right "}).status_code == 200
        got = c.get(f"/api/pipelines/{pid}").json()["pipeline"]
        assert got["keys"] == ["model"] and "sk-right" not in json.dumps(got)
        folder = settings.data / "pipelines" / pid
        assert folder.joinpath("secrets.json").stat().st_mode & 0o777 == 0o600
        c.put(f"/api/pipelines/{pid}", json={"nodes": [node("model", "model", api_key="sk-right")], "edges": []})
        assert "sk-right" not in folder.joinpath("pipeline.json").read_text()

        # The models it serves: an address without /v1 that only answers with it comes back corrected.
        found = c.post(f"/api/pipelines/{pid}/nodes/model/models", json={"base_url": fake.url[:-3] + "/"}).json()
        assert found == {"base_url": fake.url, "models": ["another", "fake-model"]}
        tried = c.post(f"/api/pipelines/{pid}/nodes/model/try", json={"message": "hello", "config": {
            "base_url": fake.url, "model": "fake-model", "thinking": "off", "temperature": 0.2}}).json()
        assert tried["reply"] == "You said: hello" and tried["completion_tokens"] == 7
        sent = fake.requests[-1]
        assert sent["chat_template_kwargs"] == {"enable_thinking": False} and sent["temperature"] == 0.2

        c.put(f"/api/pipelines/{pid}/keys/model", json={"key": "sk-wrong"})
        refused = c.post(f"/api/pipelines/{pid}/nodes/model/models", json={"base_url": fake.url})
        assert refused.status_code == 502 and refused.json()["detail"].startswith(
            "The endpoint didn't accept the API key (401). It said: Invalid API key")
        assert c.delete(f"/api/pipelines/{pid}/keys/model").status_code == 200
        assert c.get(f"/api/pipelines/{pid}").json()["pipeline"]["keys"] == []
    finally:
        fake.close()


def test_a_key_is_pasted_once_for_one_address(settings):
    fake = Fake(key="sk-right")
    try:
        c = client()
        pid = c.post("/api/pipelines", json={"name": "Two models"}).json()["pipeline"]["id"]
        c.put(f"/api/pipelines/{pid}", json={"nodes": [
            node("model", "model", base_url=fake.url, model="fake-model"),
            node("judge", "model", base_url=fake.url + "/", model="fake-model"),  # the same address, written another way
            node("other", "model", base_url="http://elsewhere:1/v1", model="m"),
        ], "edges": []})
        c.put(f"/api/pipelines/{pid}/keys/model", json={"key": "sk-right"})
        assert c.post(f"/api/pipelines/{pid}/nodes/judge/models", json={"base_url": fake.url}).status_code == 200
        tried = c.post(f"/api/pipelines/{pid}/nodes/judge/try", json={"message": "hi"})
        assert tried.status_code == 200 and tried.json()["reply"] == "You said: hi"
        folder = pipelines.folder(pid)
        nodes = pipelines.read(pid)["nodes"]
        assert pipelines.key_for(folder, nodes, "judge") == "sk-right"
        assert pipelines.key_for(folder, nodes, "other") is None  # another address doesn't borrow it
    finally:
        fake.close()


def test_an_endpoint_that_isnt_there_says_so(settings):
    c = client()
    pid = c.post("/api/pipelines", json={"name": "Nowhere"}).json()["pipeline"]["id"]
    missed = c.post(f"/api/pipelines/{pid}/nodes/model/models", json={"base_url": "http://localhost:9/v1"})
    assert missed.status_code == 502
    assert "Couldn't reach http://localhost:9/v1" in missed.json()["detail"]
    assert "use host.docker.internal for this machine" in missed.json()["detail"]


def test_replies_lose_their_thinking():
    from anvil.llm import answer_text, base_url

    assert answer_text("<think>hmm</think>\nYes.") == "Yes."
    assert answer_text("the server opened it</think> Done") == "Done"
    assert base_url(" http://h:8000/v1/chat/completions/ ") == "http://h:8000/v1"


# ---------------------------------------------------------------- runs

def test_a_run_makes_a_decision_dataset(settings, fake):
    pid = pipelines.create("Bank")["id"]
    pipelines.save(pid, labelled(fake.url))
    run_id, run, code = run_now(pid)
    assert code == 0
    st = status(run)
    assert st["state"] == "done" and st["saves"]["save"]["kept"] == 6 and st["error"] is None
    assert st["steps"]["write"]["requests"] >= 2 and st["steps"]["write"]["completion_tokens"] >= 14
    rows = kept_rows(run)
    assert len(rows) == 6
    for row in rows:
        datasets.check_decision(row["state"], row["questions"], row["gold"])
        meta, gold = json.loads(row["meta"]), json.loads(row["gold"])
        assert json.loads(row["state"]) == {"message": meta["message"]}
        assert gold == {"frustrated": meta["tone"] == "an angry"}
        assert row["source"] == f"Bank|tone={meta['tone']}"
    assert {json.loads(r["meta"])["tone"] for r in rows} <= {"an angry", "a calm"}  # the blank value isn't one

    dataset = settings.generated / f"bank-{run_id}.jsonl"
    assert dataset.read_text() == (run / "rows" / "save.jsonl").read_text()
    assert st["saves"]["save"]["dataset"] == str(dataset)
    assert datasets.detect_format(datasets.read_file(dataset).columns) == "decision"
    assert pipelines.runs(pid)[0]["id"] == run_id


def test_a_run_of_chat_rows(settings, fake):
    pid = pipelines.create("Chat")["id"]
    pipelines.save(pid, {"nodes": [
        node("m", "model", base_url=fake.url, model="fake-model"),
        node("topic", "list", column="topic", values=["cards", "loans"]),
        node("ask", "generate", column="question", prompt="Ask about {{ topic }}", per_reply=2),
        node("answer", "generate", column="answer", prompt="Answer: {{ question }}", system="Be brief."),
        node("save", "save", name="chat", rows=4, format="chat", system="You help with {{ topic }}.",
             user_column="question", assistant_column="answer"),
    ], "edges": [arrow("m", "topic"), arrow("topic", "ask"), arrow("ask", "answer"), arrow("answer", "save")]})
    _, run, code = run_now(pid)
    assert code == 0
    rows = kept_rows(run)
    assert len(rows) == 4
    for row in rows:
        system, user, assistant = row["messages"]
        topic = json.loads(row["meta"])["topic"]
        assert system == {"role": "system", "content": f"You help with {topic}."}
        assert user["content"].startswith(f"Ask about {topic} ")
        assert assistant == {"role": "assistant", "content": f"You said: Answer: {user['content']}"}
    assert any(r["messages"][0] == {"role": "system", "content": "Be brief."} for r in fake.requests)


def test_filters_drop_rows_and_say_why(settings, fake):
    graph = labelled(fake.url, wanted=4)
    graph["nodes"].insert(4, node("only", "filter", rules=[{"column": "frustrated", "op": "is", "value": "TRUE"}]))
    graph["edges"][3:] = [arrow("ask", "only"), arrow("only", "dedupe"), arrow("dedupe", "save")]
    pid = pipelines.create("Angry only")["id"]
    pipelines.save(pid, graph)
    _, run, code = run_now(pid, one_at_a_time=True)
    assert code == 0
    assert all(json.loads(r["gold"]) == {"frustrated": True} for r in kept_rows(run))
    st = status(run)
    # Angry (three kept), calm (three dropped), angry (one kept, and the other two let go: the Save is full).
    assert st["steps"]["only"]["dropped"] == 3 and st["rows_started"] == 3
    assert st["reasons"]["only"] == {"didn't pass: frustrated is 'TRUE'": 3}
    shown = [r for r in pipelines.rows(pid, run.name, "dropped", 0, 200)["rows"] if r["node"] == "only"]
    assert len(shown) == st["steps"]["only"]["dropped"]
    assert all(r["row"]["frustrated"] is False and r["reason"] == "didn't pass: frustrated is 'TRUE'" for r in shown)


def test_dedupe_drops_copies_and_a_run_that_keeps_nothing_new_fails(settings, fake, monkeypatch):
    monkeypatch.setattr(generate, "STUCK", 25)
    pid = pipelines.create("Echo")["id"]
    pipelines.save(pid, {"nodes": [
        node("m", "model", base_url=fake.url, model="fake-model", parallel=2),
        node("topic", "list", column="topic", values=["cards"]),
        node("say", "generate", column="text", prompt="Say {{ topic }}"),  # the same reply every time
        node("once", "dedupe", column="text"),
        node("save", "save", name="echo", rows=5, format="text", text_column="text"),
    ], "edges": [arrow("m", "topic"), arrow("topic", "say"), arrow("say", "once"), arrow("once", "save")]})
    _, run, code = run_now(pid)
    st = status(run)
    assert code == 1 and st["state"] == "failed" and st["saves"]["save"]["kept"] == 1
    assert st["error"] == ("The last 25 rows started didn't reach a Save. The Dedupe step dropped the most (a copy "
                           "of an earlier row). Try a preview to see what each step does.")
    assert kept_rows(run)[0]["text"] == "You said: Say cards"


def test_a_second_opinion_leaves_out_what_the_models_disagree_on(settings, fake):
    doubter = Fake(judge=lambda text: "angry" in text and text.endswith(("0", "2", "4", "6", "8")))
    try:
        graph = labelled(fake.url, wanted=8)
        graph["nodes"].append(node("second", "model", base_url=doubter.url, model="fake-model"))
        graph["edges"].append(arrow("second", "ask", "check"))
        pid = pipelines.create("Two judges")["id"]
        pipelines.save(pid, graph)
        _, run, code = run_now(pid)
        assert code == 0
        for row in kept_rows(run):
            meta, gold = json.loads(row["meta"]), json.loads(row["gold"])
            angry, even = meta["tone"] == "an angry", meta["message"].endswith(("0", "2", "4", "6", "8"))
            assert gold["frustrated"] == (False if not angry else True if even else None)
        assert doubter.requests  # the second opinion was asked
        st = status(run)
        assert set(st["blanks"]["ask"]) == {"the second opinion disagreed"}
        assert st["steps"]["ask"]["left_out"] == st["blanks"]["ask"]["the second opinion disagreed"]
    finally:
        doubter.close()


def test_answers_can_come_from_a_column_and_a_plugged_in_model_writes(settings, fake):
    writer = Fake()
    try:
        pid = pipelines.create("From the list")["id"]
        pipelines.save(pid, {"nodes": [
            node("m", "model", base_url=fake.url, model="fake-model"),
            node("w", "model", base_url=writer.url, model="fake-model"),
            node("tone", "list", column="tone", values=["frustrated", "calm", "unsure"], pick="turns"),
            node("write", "generate", column="message", prompt="Write a {{ tone }} message", per_reply=2),
            node("ask", "question", name="frustrated", instructions="Is the customer frustrated?", answer="column",
                 from_column="tone", mapping={"frustrated": "yes", "calm": "no", "unsure": "leave out"}),
            node("save", "save", name="labelled", rows=6, format="decision", state_column="message"),
        ], "edges": [arrow("m", "tone"), arrow("tone", "write"), arrow("write", "ask"), arrow("ask", "save"),
                     arrow("w", "write", "model")]})
        _, run, code = run_now(pid, one_at_a_time=True)
        assert code == 0
        rows = kept_rows(run)
        assert len(rows) == 6
        for row in rows:  # "unsure" rows have no answer at all, so they can't be saved
            tone = json.loads(row["meta"])["tone"]
            assert tone != "unsure" and json.loads(row["gold"]) == {"frustrated": tone == "frustrated"}
        assert writer.requests and not fake.requests  # the plugged-in Model wrote; the line's own wasn't asked
        st = status(run)
        assert st["reasons"]["save"] == {"none of its questions has an answer: the Question steps left them all out": 2}
        assert st["steps"]["ask"]["left_out"] == 2 and st["blanks"] == {"ask": {"left out for this value": 2}}
    finally:
        writer.close()


def test_a_save_balances_a_lists_values_by_making_only_the_ones_short(settings, fake):
    graph = labelled(fake.url, wanted=7)
    graph["nodes"][1]["config"].update(values=["an angry", "an angry", "an angry", "a calm"], pick="random")
    graph["nodes"][-1]["config"]["balance"] = "tone"
    pid = pipelines.create("Balanced")["id"]
    pipelines.save(pid, graph)
    _, run, code = run_now(pid, one_at_a_time=True)
    assert code == 0
    tones = [json.loads(r["meta"])["tone"] for r in kept_rows(run)]
    assert sorted(tones) == ["a calm"] * 3 + ["an angry"] * 4  # 7 over two values: 4 and 3, though angry is 3x as likely
    st = status(run)
    assert st["saves"]["save"]["quota"] == {"an angry": 4, "a calm": 3}
    assert st["saves"]["save"]["by_value"] == {"an angry": 4, "a calm": 3}
    assert "its tone already has its share" not in st.get("reasons", {}).get("save", {})  # none were made to throw away


def test_a_save_balances_a_questions_answers_by_dropping_the_extra(settings, fake):
    graph = labelled(fake.url, wanted=4)
    graph["nodes"][-1]["config"]["balance"] = "frustrated"
    pid = pipelines.create("Balanced answers")["id"]
    pipelines.save(pid, graph)
    _, run, code = run_now(pid, one_at_a_time=True)
    assert code == 0
    golds = [json.loads(r["gold"])["frustrated"] for r in kept_rows(run)]
    assert sorted(golds) == [False, False, True, True]
    st = status(run)
    assert st["saves"]["save"]["by_value"] == {"true": 2, "false": 2}
    # Three angry rows arrive together, and the third has no room: dropped, and counted.
    assert st["reasons"]["save"] == {"its frustrated already has its share": 1}


def test_balance_needs_a_lists_column_or_a_questions_answer(settings):
    graph = labelled("http://x:1/v1")
    graph["nodes"][-1]["config"]["balance"] = "message"
    assert [p["message"] for p in errors(graph)] == [
        "Balance by a List's column or a Question's answer: those are the columns whose values are known before the "
        "rows are made."]
    graph["nodes"][-1]["config"]["balance"] = "nope"
    assert [p["field"] for p in errors(graph)] == ["balance"]


def test_seed_rows_flow_through_and_run_out(settings, fake):
    data = settings.data / "datasets" / "uploads"
    data.mkdir(parents=True)
    (data / "messages.jsonl").write_text("".join(json.dumps({"Text": f"I am angry {i}", "id": i}) + "\n"
                                                 for i in range(5)))
    pid = pipelines.create("Label real messages")["id"]
    pipelines.save(pid, {"nodes": [
        node("m", "model", base_url=fake.url, model="fake-model"),
        node("seed", "seed", source={"kind": "file", "path": str(data / "messages.jsonl")}, columns=["Text"]),
        node("ask", "question", name="frustrated", instructions="Is the customer frustrated?", about="text"),
        node("save", "save", name="real", rows=100, format="decision", state_column="text"),
    ], "edges": [arrow("m", "seed"), arrow("seed", "ask"), arrow("ask", "save")]})
    _, run, code = run_now(pid)
    assert code == 0
    rows = kept_rows(run)
    assert sorted(json.loads(r["state"])["message"] for r in rows) == [f"I am angry {i}" for i in range(5)]
    assert all(json.loads(r["gold"]) == {"frustrated": True} for r in rows)
    st = status(run)
    assert st["state"] == "done" and st["saves"]["save"]["closed"] and st["seeds"] == {"seed": 5}


def stop_after(run, rows):
    """Press Stop once a Save has this many rows."""
    def wait():
        file = run / "rows" / "save.jsonl"
        while not file.exists() or len(file.read_text().splitlines()) < rows:
            time.sleep(0.02)
        (run / "stop.json").write_text("{}")
    threading.Thread(target=wait, daemon=True).start()


def test_stop_then_resume_carries_on(settings):
    slow = Fake(delay=0.04)
    try:
        pid = pipelines.create("Slow")["id"]
        pipelines.save(pid, labelled(slow.url, wanted=60, parallel=1))
        run_id, run = pipelines.new_run(pid, False)
        stop_after(run, 3)
        assert generate.Run(pipelines.folder(pid), run).run() == 0
        first = status(run)
        assert first["state"] == "stopped" and 3 <= first["saves"]["save"]["kept"] < 60
        before = kept_rows(run)
        assert (settings.generated / f"bank-{run_id}.jsonl").exists()  # what a stopped run kept is a dataset already

        (run / "stop.json").unlink()
        assert generate.Run(pipelines.folder(pid), run).run() == 0
        after = status(run)
        assert after["state"] == "done" and after["resumed"] == 1 and after["saves"]["save"]["kept"] == 60
        rows = kept_rows(run)
        assert rows[:len(before)] == before and len({r["state"] for r in rows}) == 60
        assert after["steps"]["write"]["requests"] > first["steps"]["write"]["requests"]
        assert len((settings.generated / f"bank-{run_id}.jsonl").read_text().splitlines()) == 60
    finally:
        slow.close()


def test_a_preview_shows_what_each_step_did(settings, fake):
    pid = pipelines.create("Preview")["id"]
    pipelines.save(pid, labelled(fake.url))
    _, run, code = run_now(pid, preview=True)
    assert code == 0
    entries = pipelines.rows(pid, "preview", "preview", 0, 200)["rows"]
    kept = [e for e in entries if e["outcome"] == "kept"]
    assert len(kept) == generate.PREVIEW_ROWS
    steps = kept[0]["trace"]
    assert [s["node"] for s in steps] == ["tone", "write", "ask", "dedupe", "save"]
    assert steps[1]["asked"][0]["prompt"].startswith("Write a") and "Is the customer frustrated?" in \
        steps[2]["asked"][0]["prompt"]
    assert steps[2]["answer"] == json.loads(kept[0]["record"]["gold"])["frustrated"]
    assert not settings.generated.exists() or not any(settings.generated.iterdir())  # a preview makes no dataset


def test_a_pipeline_deleted_while_listed_is_left_out(settings, monkeypatch):
    kept, gone = pipelines.create("Kept")["id"], pipelines.create("Gone")["id"]
    runs = pipelines.runs

    def deleted_meanwhile(pipeline_id):
        if pipeline_id == gone:  # the delete lands after its pipeline.json was read, before its runs are
            shutil.rmtree(pipelines.folder(gone))
        return runs(pipeline_id)

    monkeypatch.setattr(pipelines, "runs", deleted_meanwhile)
    assert [p["id"] for p in pipelines.list_pipelines()] == [kept]


def test_the_server_runs_pipelines_in_their_own_process(settings, fake):
    c = client()
    pid = c.post("/api/pipelines", json={"name": "Spawned"}).json()["pipeline"]["id"]
    assert c.post(f"/api/pipelines/{pid}/runs", json={"preview": True}).status_code == 409  # nothing to save yet
    saved = c.put(f"/api/pipelines/{pid}", json={"name": "Spawned", **labelled(fake.url)}).json()
    assert [p for p in saved["problems"] if p["level"] == "error"] == []
    assert c.post(f"/api/pipelines/{pid}/runs", json={"preview": True}).json() == {"id": "preview"}
    deadline = time.time() + 60
    while (run := c.get(f"/api/pipelines/{pid}/runs/preview").json()["run"])["state"] in ("starting", "running"):
        assert time.time() < deadline
        time.sleep(0.25)
    assert run["state"] == "done", run
    shown = c.get(f"/api/pipelines/{pid}/runs/preview/rows", params={"which": "preview"}).json()
    assert sum(e["outcome"] == "kept" for e in shown["rows"]) == generate.PREVIEW_ROWS
    assert c.get(f"/api/pipelines/{pid}").json()["pipeline"]["preview"]["state"] == "done"
    assert c.get("/api/pipelines").json()["pipelines"][0]["id"] == pid
    assert c.delete(f"/api/pipelines/{pid}").json() == {"ok": True}
    assert c.get(f"/api/pipelines/{pid}").status_code == 404
