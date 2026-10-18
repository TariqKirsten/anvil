"""Logging in, because anvil listens on the network.

A login sets a signed cookie that names the account and lasts 30 days. Every /api route except the ones in OPEN needs
it. The signing key mixes a random secret kept in ANVIL_DATA with the account's password hash, so changing the
password logs that account out everywhere else.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from . import accounts, config

COOKIE = "anvil_session"
MAX_AGE = 30 * 24 * 3600
OPEN = {"/api/health", "/api/session", "/api/login", "/api/logout"}

router = APIRouter()


def _secret() -> bytes:
    path = config.settings.data / "session_secret"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_bytes()


def _signature(user: accounts.User, expires: str) -> str:
    key = hashlib.sha256(_secret() + user.password_hash.encode()).digest()
    return hmac.new(key, f"{user.id}.{expires}".encode(), hashlib.sha256).hexdigest()


def make_token(user: accounts.User, now: float | None = None) -> str:
    expires = str(int((now if now is not None else time.time()) + MAX_AGE))
    return f"{user.id}.{expires}.{_signature(user, expires)}"


def user_from_token(token: str | None, now: float | None = None) -> accounts.User | None:
    parts = (token or "").split(".")
    if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    if int(parts[1]) <= (now if now is not None else time.time()):
        return None
    user = accounts.get(int(parts[0]))
    return user if user and hmac.compare_digest(parts[2], _signature(user, parts[1])) else None


def current_user(request: Request) -> accounts.User | None:
    return user_from_token(request.cookies.get(COOKIE))


def set_cookie(response: Response, user: accounts.User) -> None:
    response.set_cookie(COOKIE, make_token(user), max_age=MAX_AGE, httponly=True, samesite="lax")


class Login(BaseModel):
    username: str = accounts.DEFAULT_USER
    password: str


@router.get("/api/session")
def session(request: Request) -> dict:
    user = current_user(request)
    return {"logged_in": user is not None, "username": user.username if user else None,
            "has_accounts": accounts.any_users()}


@router.post("/api/login")
def login(body: Login, response: Response) -> dict:
    if not accounts.any_users():
        raise HTTPException(503, "There's no account yet. Set ANVIL_PASSWORD in .env, then restart anvil.")
    user = accounts.authenticate(body.username.strip(), body.password)
    if user is None:
        time.sleep(1)  # slows down guessing
        raise HTTPException(401, "That username or password isn't right.")
    set_cookie(response, user)
    return {"logged_in": True, "username": user.username}


@router.post("/api/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE)
    return {"logged_in": False}
