import time

from fastapi.testclient import TestClient

from anvil import accounts, auth
from anvil.main import app
from conftest import write_model


def client():
    return TestClient(app)  # not as a context manager, so the GPU probe at startup doesn't run


def login(c, password="correct horse", username="admin"):
    return c.post("/api/login", json={"username": username, "password": password})


def test_api_needs_login(settings):
    c = client()
    assert c.get("/api/folders").status_code == 401
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/session").json() == {"logged_in": False, "username": None, "has_accounts": True}


def test_wrong_password_or_user_is_refused(settings):
    c = client()
    assert login(c, "wrong").status_code == 401
    assert login(c, username="nobody").status_code == 401
    assert c.get("/api/folders").status_code == 401


def test_login_opens_the_api(settings):
    c = client()
    assert login(c).status_code == 200
    assert c.get("/api/session").json()["username"] == "admin"
    assert c.get("/api/folders").json()["roots"] == [str(settings.model_roots[0])]


def test_tampered_or_expired_tokens_fail(settings):
    user = accounts.authenticate("admin", "correct horse")
    token = auth.make_token(user)
    assert auth.user_from_token(token) == user
    assert auth.user_from_token(token[:-1] + ("0" if token[-1] != "0" else "1")) is None
    assert auth.user_from_token(auth.make_token(user, now=time.time() - auth.MAX_AGE - 10)) is None
    assert auth.user_from_token("2" + token[1:]) is None  # another account's id with this signature


def test_folders_stay_inside_the_model_roots(settings, tmp_path):
    c = client()
    login(c)
    assert c.get("/api/folders", params={"path": str(tmp_path)}).status_code == 403
    assert c.post("/api/folders/check", json={"path": "/etc"}).status_code == 403
    model = write_model(settings.model_roots[0] / "tiny")
    report = c.post("/api/folders/check", json={"path": str(model)}).json()
    assert report["verdict"] == "ok"


def test_screen_never_serves_files_outside_the_build(tmp_path, monkeypatch):
    from anvil import config
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("the screen")
    (dist / "assets" / "app.js").write_text("code")
    (tmp_path / "secret.txt").write_text("secret")
    monkeypatch.setattr(config, "settings", config.Settings(password="x", data=tmp_path, web_dist=dist))
    c = client()
    assert c.get("/assets/app.js").text == "code"
    assert c.get("/runs/42").text == "the screen"
    for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/..%2f..%2fsecret.txt"):
        assert c.get(path).text != "secret", path


def test_no_account_means_nobody_can_log_in(tmp_path, monkeypatch):
    from anvil import config
    monkeypatch.setattr(config, "settings", config.Settings(password="", data=tmp_path, model_roots=()))
    accounts.ensure_default_user()
    c = client()
    assert login(c, "").status_code == 503
    assert c.get("/api/folders").status_code == 401


def test_a_run_is_deleted_through_the_api(settings):
    c = client()
    login(c)
    run = settings.data / "runs" / "old-run"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}")
    (run / "status.json").write_text('{"state": "done"}')
    answer = c.delete("/api/runs/old-run")
    assert answer.status_code == 200 and answer.json()["freed"] > 0 and not run.exists()
    assert c.delete("/api/runs/old-run").status_code == 404
    assert c.delete("/api/runs/Not_A_Run").status_code == 404
