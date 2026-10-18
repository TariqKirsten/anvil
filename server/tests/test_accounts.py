from fastapi.testclient import TestClient

from anvil import accounts, auth, config, main


def logged_in():
    c = TestClient(main.app)
    assert c.post("/api/login", json={"username": "admin", "password": "correct horse"}).status_code == 200
    return c


def test_admin_is_created_once_from_the_env_password(settings, monkeypatch):
    assert accounts.authenticate("admin", "correct horse")
    # A later .env password doesn't change the account: the app owns the password now.
    monkeypatch.setattr(config, "settings", config.Settings(password="something else", data=settings.data))
    accounts.ensure_default_user()
    assert accounts.authenticate("admin", "correct horse")
    assert not accounts.authenticate("admin", "something else")


def test_reset_sets_the_password_back_to_the_env_one(settings, monkeypatch):
    monkeypatch.setattr(config, "settings",
                        config.Settings(password="new one!", data=settings.data, reset_password=True))
    accounts.ensure_default_user()
    assert accounts.authenticate("admin", "new one!")


def test_passwords_are_stored_hashed(settings):
    user = accounts.authenticate("admin", "correct horse")
    assert user.password_hash.startswith("scrypt$") and "correct horse" not in user.password_hash


def test_changing_the_password_logs_out_old_sessions(settings):
    c = logged_in()
    old_token = c.cookies.get(auth.COOKIE)
    assert c.put("/api/account/password", json={"current": "wrong", "new": "longer password"}).status_code == 400
    assert c.put("/api/account/password", json={"current": "correct horse", "new": "short"}).status_code == 400
    assert c.put("/api/account/password", json={"current": "correct horse", "new": "longer password"}).status_code == 200
    assert auth.user_from_token(old_token) is None
    assert c.get("/api/account").status_code == 200  # this browser got a fresh cookie
    assert accounts.authenticate("admin", "longer password")


def test_hf_token_is_checked_saved_and_never_sent_back(settings, monkeypatch):
    monkeypatch.setattr(main, "hub_whoami", lambda token: {"name": "someone"})
    c = logged_in()
    assert c.get("/api/account").json()["hf"] == {"connected": False, "user": None, "source": None}
    assert c.put("/api/account/hf-token", json={"token": "not-a-token"}).status_code == 400
    saved = c.put("/api/account/hf-token", json={"token": " hf_abc123 "})
    assert saved.json()["hf"] == {"connected": True, "user": "someone", "source": "account"}
    assert "hf_abc123" not in c.get("/api/account").text
    assert accounts.authenticate("admin", "correct horse").hf_token == "hf_abc123"
    assert c.delete("/api/account/hf-token").json()["hf"]["connected"] is False


def test_rejected_hf_token_is_not_saved(settings, monkeypatch):
    import httpx2
    from huggingface_hub.errors import HfHubHTTPError

    def whoami(token):  # what the Hub answers for a bad token (checked against the real Hub)
        rejected = httpx2.Response(401, request=httpx2.Request("GET", "https://huggingface.co/api/whoami-v2"))
        raise HfHubHTTPError("Invalid user token.", response=rejected)

    monkeypatch.setattr(main, "hub_whoami", whoami)
    c = logged_in()
    response = c.put("/api/account/hf-token", json={"token": "hf_wrong"})
    assert response.status_code == 400 and "didn't accept" in response.json()["detail"]
    assert accounts.authenticate("admin", "correct horse").hf_token is None


def test_env_token_is_the_fallback(settings, monkeypatch):
    monkeypatch.setattr(config, "settings", config.Settings(password="correct horse", data=settings.data,
                                                            hf_token="hf_from_env"))
    c = logged_in()
    assert c.get("/api/account").json()["hf"] == {"connected": True, "user": None, "source": "env"}
