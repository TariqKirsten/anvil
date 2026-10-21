"""Uploading a run to Hugging Face, against a stand-in for the Hub that keeps repos in memory: what goes to which repo,
the cards, visibility, and the refusals. Nothing here talks to Hugging Face."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from fastapi.testclient import TestClient
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from anvil import accounts, config, main, runs, upload
from test_api import login

FINAL_TEST = {"step": 52000, "rows": 5000, "overall": 0.8708, "accuracy": 0.9536, "auroc": 0.9897, "ece": 0.0195,
              "questions": {"abuse_severity": {"type": "score", "rows": 129, "mae": 0.2403},
                            "jailbreak": {"type": "noul", "rows": 5000, "auroc": 0.9981}}}


def fake_run(root: Path, name="decision-0-8b") -> Path:
    """A finished LoRA Clef run's folder, as far as an upload reads it: its model, its adapter and a GGUF export."""
    run = root / name
    merged, adapter, gguf = run / "model" / "merged", run / "model" / "adapter", run / "exports" / "final"
    for folder in (merged, adapter, gguf):
        folder.mkdir(parents=True)
    for folder in (merged, adapter):
        (folder / "joint_head_config.json").write_text("{}")
        (folder / "joint_head.safetensors").write_bytes(b"head")
        (folder / "anvil.json").write_text(json.dumps({"base": "/models/local", "run": name, "picked_step": 52000,
                                                       "final_test": FINAL_TEST}))
        (folder / "README.md").write_text("PEFT's own template")
    (merged / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}))
    (merged / "model.safetensors").write_bytes(b"w" * 1000)
    (merged / "tokenizer.json").write_text("{}")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"a" * 100)
    files = {"decision-0-8b-final-Q8_0.gguf": 600, "decision-0-8b-final-Q4_K_M.gguf": 400}
    for file, size in files.items():
        (gguf / file).write_bytes(b"g" * size)
    (gguf / "status.json").write_text(json.dumps({"state": "done", "files": [{"name": f, "size": s} for f, s in files.items()]}))
    (run / "config.json").write_text(json.dumps({
        "id": name, "name": "decision-0.8B", "train_as": "decision", "method": "clef", "training_type": "lora",
        "model": {"source": "hub", "id": "Qwen/Qwen3.5-0.8B", "kind": "model", "base_model": None},
        "data": {"prepared": str(root / "prepared"), "format": "decision",
                 "rows": {"train": 1000, "validation": 50, "test": 50}},
        "gpus": [1], "gpu": 1,
        "hyper": {"learningRate": "5e-5", "headLearningRate": "1e-4", "epochs": 1, "batchSize": 8, "loraRank": 32,
                  "loraAlpha": 64, "warmupPercent": 3, "scheduler": "cosine", "maxLength": 2048, "precision": "bf16"},
        "checkpoints": {"every": 500, "keepBest": 3, "pickBy": "overall"}}))
    (run / "status.json").write_text(json.dumps({"state": "done", "step": 66636, "total": 66636, "items_per_step": 16}))
    (run / "leaderboard.json").write_text(json.dumps({"entries": [
        {"step": 52000, "score": 0.9363, "metric": "overall", "direction": "max"}], "best_step": 52000}))
    return run


def missing(repo: str) -> RepositoryNotFoundError:
    request = httpx2.Request("GET", f"https://huggingface.co/api/models/{repo}")
    return RepositoryNotFoundError("missing", response=httpx2.Response(404, request=request))


class FakeHub:
    """The Hub, in memory: repo -> {"private", "files": name -> bytes}."""

    def __init__(self, repos=None):
        self.repos, self.calls = repos or {}, []

    def repo_info(self, repo):
        if repo not in self.repos:
            raise missing(repo)
        found = self.repos[repo]
        return SimpleNamespace(private=found["private"], siblings=[SimpleNamespace(rfilename=n) for n in found["files"]])

    def create_repo(self, repo, visibility, exist_ok):
        self.calls.append(("create", repo, visibility))
        self.repos[repo] = {"private": visibility == "private", "files": {}}

    def update_repo_settings(self, repo, visibility):
        self.calls.append(("visibility", repo, visibility))
        self.repos[repo]["private"] = visibility == "private"

    def hf_hub_download(self, repo, filename, cache_dir):
        path = Path(cache_dir) / repo.replace("/", "--") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.repos[repo]["files"][filename])
        return str(path)

    def create_commit(self, repo, operations, commit_message, commit_description):
        sent = {op.path_in_repo: op.path_or_fileobj if isinstance(op.path_or_fileobj, bytes)
                else Path(op.path_or_fileobj).read_bytes() for op in operations}
        self.repos[repo]["files"].update(sent)
        self.calls.append(("commit", repo, sorted(sent)))
        return SimpleNamespace(commit_url=f"https://huggingface.co/{repo}/commit/1")


def ask(run: Path, **want) -> tuple[str, upload.Status]:
    upload_id = "20261021-120000"
    folder = run / "uploads" / upload_id
    folder.mkdir(parents=True)
    (folder / "request.json").write_text(json.dumps({"version": "final", "private": True, **want}))
    return upload_id, upload.Status(folder)


def test_the_weights_and_their_gguf_files_go_up_to_repos_of_their_own_with_cards(tmp_path):
    run = fake_run(tmp_path)
    hub = FakeHub()
    upload_id, status = ask(run, weights="merged", repo="me/decision-0.8B", gguf_repo="me/decision-0.8B-GGUF",
                            gguf=["decision-0-8b-final-Q8_0.gguf", "decision-0-8b-final-Q4_K_M.gguf"])
    upload.upload(run, upload_id, status, hub)

    assert hub.calls[0] == ("create", "me/decision-0.8B", "private")
    weights = hub.repos["me/decision-0.8B"]["files"]
    # anvil.json holds this machine's paths and PEFT's README is a blank template: the card stands in for both.
    assert sorted(weights) == ["README.md", "config.json", "joint_head.safetensors", "joint_head_config.json",
                               "model.safetensors", "tokenizer.json"]
    assert weights["model.safetensors"] == b"w" * 1000
    card = weights["README.md"].decode()
    assert card.startswith("---\nbase_model: Qwen/Qwen3.5-0.8B\nbase_model_relation: finetune\n"
                           "library_name: transformers\n")
    assert "| Overall | 0.871 |" in card and "| `jailbreak` | yes/no | AUROC 0.998 | 5,000 |" in card
    assert "picked by its overall score on validation, scored once on 5,000 rows of the test set" in card
    assert 'load_release_model("me/decision-0.8B")' in card and "LoRA (rank 32, alpha 64)" in card
    assert "/models/local" not in card and str(tmp_path) not in card and upload.ours(card)

    gguf = hub.repos["me/decision-0.8B-GGUF"]
    assert gguf["private"] and sorted(gguf["files"]) == [
        "README.md", "decision-0-8b-final-Q4_K_M.gguf", "decision-0-8b-final-Q8_0.gguf"]
    card = gguf["files"]["README.md"].decode()
    assert "base_model: me/decision-0.8B\nbase_model_relation: quantized\nlibrary_name: gguf\n" in card
    assert "llama-server -hf me/decision-0.8B-GGUF:Q8_0 -c 8192 -b 8192 -ub 8192" in card and "/v1/systemone" in card

    done = json.loads(status.path.read_text())
    sent = sum(len(content) for repo in hub.repos.values() for name, content in repo["files"].items() if name != "README.md")
    assert done["state"] == "done" and done["bytes_done"] == done["bytes_total"] == sent
    assert [(r["id"], r["state"], r["created"], r["card"]) for r in done["repos"]] == [
        ("me/decision-0.8B", "done", True, "written"), ("me/decision-0.8B-GGUF", "done", True, "written")]


def test_an_existing_repo_takes_the_visibility_picked_and_keeps_a_card_someone_edited(tmp_path):
    run = fake_run(tmp_path)
    hub = FakeHub({"me/decision": {"private": True, "files": {"README.md": b"# My own words\n"}}})
    upload_id, status = ask(run, weights="merged", repo="me/decision", gguf=["decision-0-8b-final-Q8_0.gguf"],
                            gguf_repo="me/decision", private=False)  # one repo for both
    upload.upload(run, upload_id, status, hub)
    assert hub.calls == [("visibility", "me/decision", "public"), ("commit", "me/decision", [
        "config.json", "decision-0-8b-final-Q8_0.gguf", "joint_head.safetensors", "joint_head_config.json",
        "model.safetensors", "tokenizer.json"])]
    assert hub.repos["me/decision"]["files"]["README.md"] == b"# My own words\n"
    repo = json.loads(status.path.read_text())["repos"][0]
    assert repo["made"] == "public" and repo["card"] == "kept" and not repo["created"]

    # A card anvil wrote, untouched since, is anvil's to replace with the new one.
    hub.repos["me/decision"]["files"]["README.md"] = upload.card(
        json.loads((run / "config.json").read_text()), run, "final", "merged", "me/decision", []).encode()
    folder = run / "uploads" / "20261021-120001"
    folder.mkdir()
    (folder / "request.json").write_text(json.dumps({"version": "final", "weights": "merged", "repo": "me/decision",
                                                     "private": False}))
    upload.upload(run, folder.name, upload.Status(folder), hub)
    assert "README.md" in hub.calls[-1][2]


def test_a_lora_adapter_goes_up_alone_with_its_head(tmp_path):
    run = fake_run(tmp_path)
    hub = FakeHub()
    upload_id, status = ask(run, weights="adapter", repo="me/decision-lora")
    upload.upload(run, upload_id, status, hub)
    files = hub.repos["me/decision-lora"]["files"]
    assert sorted(files) == ["README.md", "adapter_config.json", "adapter_model.safetensors",
                             "joint_head.safetensors", "joint_head_config.json"]
    card = files["README.md"].decode()
    assert "base_model_relation: adapter\nlibrary_name: peft\n" in card
    assert 'PeftModel.from_pretrained(base, "me/decision-lora")' in card


def test_an_edited_card_is_no_longer_anvils():
    text = "---\nlibrary_name: gguf\n---\n\n# A model\n"
    marked = text + f"\n<!-- anvil {__import__('hashlib').sha256(text.encode()).hexdigest()[:16]} -->\n"
    assert upload.ours(marked)
    assert not upload.ours(marked.replace("# A model", "# A better model"))
    assert not upload.ours(text)


def test_a_refused_upload_says_the_token_cant_write_there(tmp_path, monkeypatch):
    import huggingface_hub

    run = fake_run(tmp_path)
    upload_id, status = ask(run, weights="merged", repo="someone-else/decision")

    class Refusing(FakeHub):
        def __init__(self, token=None):
            super().__init__()

        def create_commit(self, repo, **kwargs):
            request = httpx2.Request("POST", f"https://huggingface.co/api/models/{repo}/commit/main")
            raise HfHubHTTPError("Forbidden", response=httpx2.Response(403, request=request))

    monkeypatch.setattr(huggingface_hub, "HfApi", Refusing)
    monkeypatch.setattr(upload, "_watch_progress", lambda: None)  # leave the library's progress bars be
    monkeypatch.setattr(sys, "argv", ["anvil.upload", str(run), upload_id])
    with pytest.raises(SystemExit):
        upload.main()
    failed = json.loads(status.path.read_text())
    assert failed["state"] == "failed" and "can't write there" in failed["error"]
    assert failed["repos"][0]["state"] == "failed"


# ---------------------------------------------------------------- the server's side: starting, refusing, cancelling

class Recorded:
    """A stand-in for the upload process: still going until told otherwise."""

    def __init__(self, args, **kwargs):
        self.args, self.env, self.kwargs, self.pid, self.code = args, kwargs["env"], kwargs, 999_999_999, None
        Recorded.last = self

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "settings", config.Settings(password="x", data=tmp_path / "data", hf_token=None))
    monkeypatch.setattr(runs, "_processes", {})
    return fake_run(tmp_path / "data" / "runs")


def test_starting_an_upload_checks_what_it_was_asked_and_hands_the_token_on(served, monkeypatch):
    monkeypatch.setattr(runs.subprocess, "Popen", Recorded)
    want = {"version": "final", "weights": "merged", "repo": "me/decision", "private": True}
    with pytest.raises(ValueError, match="Connect your Hugging Face account"):
        runs.start_upload("decision-0-8b", want, None)
    for bad, why in (({"version": "step-0001234"}, "doesn't exist"),
                     ({"gguf": ["other.gguf"], "gguf_repo": "me/x"}, "aren't in this version's export"),
                     ({"weights": None}, "Pick the weights, a GGUF file, or both"),
                     ({"weights": "everything"}, "merged"),
                     ({"repo": "not a repo"}, "your-name/model-name, not not a repo")):
        with pytest.raises(ValueError, match=why):
            runs.start_upload("decision-0-8b", {**want, **bad}, "hf_token")
    upload_id = runs.start_upload("decision-0-8b", {**want, "gguf": ["decision-0-8b-final-Q8_0.gguf"],
                                                    "gguf_repo": "me/decision-GGUF"}, "hf_secret")
    sent = Recorded.last
    assert sent.args[1:] == ["-m", "anvil.upload", str(served), upload_id] and sent.kwargs["start_new_session"]
    assert sent.env["HF_TOKEN"] == "hf_secret" and sent.env["HIP_VISIBLE_DEVICES"] == ""
    request = json.loads((served / "uploads" / upload_id / "request.json").read_text())
    assert request["gguf_repo"] == "me/decision-GGUF" and request["private"] is True
    assert "hf_secret" not in (served / "uploads" / upload_id / "request.json").read_text()
    listed = runs.uploads("decision-0-8b")
    assert listed[0]["id"] == upload_id and listed[0]["state"] == "starting" and listed[0]["repo"] == "me/decision"

    with pytest.raises(ValueError, match="still going"):
        runs.start_upload("decision-0-8b", want, "hf_token")
    with pytest.raises(ValueError, match="being uploaded"):
        runs.start_export("decision-0-8b", "final", ["Q8_0"])
    with pytest.raises(ValueError, match="being uploaded"):
        runs.delete("decision-0-8b")
    sent.code = 0  # it has finished without saying so: it shows as failed
    assert runs.uploads("decision-0-8b")[0]["state"] == "failed"


def test_cancel_ends_an_upload_outright(served):
    folder = served / "uploads" / "20261021-130000"
    folder.mkdir(parents=True)
    (folder / "status.json").write_text(json.dumps({"state": "running", "stage": "Uploading to me/decision"}))
    stand_in = subprocess.Popen(["sleep", "60"], start_new_session=True)
    runs._processes[f"decision-0-8b:upload:{folder.name}"] = stand_in
    runs.cancel_upload("decision-0-8b", folder.name)
    assert stand_in.poll() is not None and json.loads((folder / "status.json").read_text())["state"] == "cancelled"
    with pytest.raises(ValueError, match="isn't going"):
        runs.cancel_upload("decision-0-8b", folder.name)
    with pytest.raises(FileNotFoundError):
        runs.cancel_upload("decision-0-8b", "../../etc")


def test_the_screen_learns_where_the_token_can_upload(settings, monkeypatch):
    c = TestClient(main.app)
    login(c)
    assert c.get("/api/hub/me").json() == {"connected": False, "name": None, "namespaces": [], "write": False}
    user = accounts.authenticate("admin", "correct horse")
    accounts.set_hf_token(user.id, "hf_write", "me")
    answers = {
        "hf_write": {"name": "me", "orgs": [{"name": "my-lab"}], "auth": {"accessToken": {"role": "write"}}},
        "hf_read": {"name": "me", "orgs": [{"name": "my-lab"}], "auth": {"accessToken": {"role": "read"}}},
        "hf_fine": {"name": "me", "orgs": [{"name": "my-lab"}], "auth": {"accessToken": {
            "role": "fineGrained", "fineGrained": {"scoped": [
                {"entity": {"type": "user", "name": "me"}, "permissions": ["repo.content.read"]},
                {"entity": {"type": "org", "name": "my-lab"}, "permissions": ["repo.content.read", "repo.write"]}]}}}},
    }
    monkeypatch.setattr(main, "hub_whoami", lambda token: answers[token])
    monkeypatch.setattr(main, "_accounts", {})
    assert c.get("/api/hub/me").json() == {"connected": True, "name": "me", "namespaces": ["me", "my-lab"],
                                           "write": True}
    for token, places in (("hf_read", []), ("hf_fine", ["my-lab"])):
        accounts.set_hf_token(user.id, token, "me")
        assert c.get("/api/hub/me").json()["namespaces"] == places
