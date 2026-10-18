"""User accounts, kept in a small SQLite database in ANVIL_DATA.

anvil starts with one account, admin. It's created on the first start with the password from ANVIL_PASSWORD in .env.
From then on the password lives here, hashed, and is changed in the app. ANVIL_RESET_PASSWORD=1 sets it back to
ANVIL_PASSWORD, for when it's forgotten.

Each account keeps its own Hugging Face token. The screen never gets the token back, only whose account it is.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass

from . import config

DEFAULT_USER = "admin"
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    hf_token TEXT,
    hf_user TEXT,
    created_at REAL NOT NULL
)
"""


@dataclass(frozen=True)
class User:
    id: int
    username: str
    password_hash: str
    hf_token: str | None
    hf_user: str | None


def _connect() -> sqlite3.Connection:
    path = config.settings.data / "anvil.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    conn = sqlite3.connect(path)
    if fresh:
        path.chmod(0o600)  # it holds Hugging Face tokens
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def _user(row: sqlite3.Row | None) -> User | None:
    if row is None:
        return None
    return User(row["id"], row["username"], row["password_hash"], row["hf_token"], row["hf_user"])


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, salt, _ = stored.split("$")
        return hmac.compare_digest(hash_password(password, bytes.fromhex(salt)), stored)
    except ValueError:
        return False


def any_users() -> bool:
    with closing(_connect()) as conn:
        return conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def get(user_id: int) -> User | None:
    with closing(_connect()) as conn:
        return _user(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


def authenticate(username: str, password: str) -> User | None:
    with closing(_connect()) as conn:
        user = _user(conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone())
    return user if user and check_password(password, user.password_hash) else None


def _update(user_id: int, **fields) -> User:
    columns = ", ".join(f"{name} = ?" for name in fields)
    with closing(_connect()) as conn, conn:
        conn.execute(f"UPDATE users SET {columns} WHERE id = ?", (*fields.values(), user_id))
    user = get(user_id)
    assert user is not None
    return user


def set_password(user_id: int, password: str) -> User:
    return _update(user_id, password_hash=hash_password(password))


def set_hf_token(user_id: int, token: str | None, hf_user: str | None) -> User:
    return _update(user_id, hf_token=token, hf_user=hf_user)


def ensure_default_user() -> None:
    """Create admin on the first start, and reset its password when ANVIL_RESET_PASSWORD=1."""
    password = config.settings.password
    if not password:
        return
    with closing(_connect()) as conn, conn:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (DEFAULT_USER,)).fetchone()
        if row is None and conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None:
            conn.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                         (DEFAULT_USER, hash_password(password), time.time()))
        elif row is not None and config.settings.reset_password:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(password), row["id"]))
