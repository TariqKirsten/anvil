"""GPUs shared with a model server: the address anvil calls before training (and its key), what happens when it
refuses or isn't there, and pipelines and the assistant holding back from that server while training runs."""

import json
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from anvil import assistant, pipelines, runs, sharing
from test_pipelines import client, labelled
from test_training import FORM, run_settings  # noqa: F401 - run_settings is a fixture


class Server:
    """Something listening where the address points: it answers status to every POST, and remembers each one."""

    def __init__(self, status=200):
        self.calls: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                server.calls.append({"path": self.path, "auth": self.headers.get("Authorization")})
                body = json.dumps({"msg": "ok" if status < 300 else "it broke"}).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.http.server_port}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def close(self):
        self.http.shutdown()


@pytest.fixture
def server():
    made = []

    def make(status=200):
        made.append(Server(status))
        return made[-1]

    yield make
    for s in made:
        s.close()


def test_the_address_and_key_stay_on_the_server(settings):
    c = client()
    assert c.get("/api/sharing").json()["sharing"] == {"url": "", "has_key": False}
    assert c.put("/api/sharing", json={"url": "host:8000/unload"}).status_code == 400
    saved = c.put("/api/sharing", json={"url": "http://host.docker.internal:8000/api/models/unload"}).json()
    assert saved["sharing"]["url"] == "http://host.docker.internal:8000/api/models/unload"
    assert c.put("/api/sharing/key", json={"key": "sk-unload"}).json()["sharing"]["has_key"] is True
    assert stat.S_IMODE((settings.data / "sharing-secret.json").stat().st_mode) == 0o600
    assert "sk-unload" not in c.get("/api/sharing").text
    assert c.delete("/api/sharing/key").json()["sharing"]["has_key"] is False


def test_freeing_the_cards_calls_the_address_with_the_right_key(settings, server):
    up = server()
    sharing.save(f"{up.base}/api/models/unload")
    assistant.save_settings({"base_url": f"{up.base}/v1", "model": "helper"})
    assistant.set_key("sk-assistant")
    assert sharing.free_cards().startswith("Asked")
    assert up.calls == [{"path": "/api/models/unload", "auth": "Bearer sk-assistant"}]  # the assistant's server
    sharing.set_key("sk-own")
    sharing.free_cards()
    assert up.calls[-1]["auth"] == "Bearer sk-own"

    sharing.set_key(None)
    assistant.save_settings({"base_url": "http://elsewhere:9/v1"})
    sharing.free_cards()
    assert up.calls[-1]["auth"] is None  # another server's key isn't sent

    broken = server(status=500)
    sharing.save(f"{broken.base}/unload")
    with pytest.raises(ValueError, match="answered 500"):
        sharing.free_cards()
    assert client().post("/api/sharing/free").status_code == 409

    sharing.save("http://127.0.0.1:9/unload")  # nothing listens there: nothing holds the cards
    assert sharing.free_cards().startswith("Nothing answered")
    sharing.save("")
    assert sharing.free_cards() is None


def test_training_frees_the_cards_first(run_settings, server):  # noqa: F811 - the fixture
    up = server()
    sharing.save(f"{up.base}/unload")
    run_id = runs.create(FORM, {"train": {}}, None)
    assert len(up.calls) == 1 and run_settings[0][0] == run_id

    broken = server(status=503)
    sharing.save(f"{broken.base}/unload")
    with pytest.raises(ValueError, match="Couldn't free the GPUs"):
        runs.create(FORM, {"train": {}}, None)
    assert len(run_settings) == 1  # no run started on cards still held


def test_while_training_nothing_asks_that_servers_models(settings, monkeypatch):
    sharing.save("http://models:8000/api/models/unload")
    monkeypatch.setattr(runs, "busy_gpus", lambda: {0: "big-run"})
    here = pipelines.create("Here")["id"]
    pipelines.save(here, labelled("http://models:8000/v1"))
    with pytest.raises(ValueError, match="big-run is training"):
        pipelines.new_run(here, True)
    elsewhere = pipelines.create("Elsewhere")["id"]
    pipelines.save(elsewhere, labelled("http://other:8000/v1"))
    assert pipelines.new_run(elsewhere, True)[0] == "preview"

    assistant.save_settings({"base_url": "http://models:8000/v1", "model": "helper"})
    assert assistant.busy(assistant.settings()) == {"training": "big-run", "base_url": "http://models:8000/v1"}
    monkeypatch.setattr(runs, "busy_gpus", lambda: {})
    assert assistant.busy(assistant.settings()) is None
