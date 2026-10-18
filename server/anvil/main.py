"""The anvil server: the API under /api, and the built screen at every other path.

Run it with docker/dev.sh (development) or docker/app.sh (the finished app), inside the container.
"""

from __future__ import annotations

import re
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from . import accounts, auth, config, folders, gpus, models_check

REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts.ensure_default_user()
    # The first PyTorch probe takes a few seconds, so start it now instead of on the first page load.
    threading.Thread(target=gpus.probe, daemon=True).start()
    yield


app = FastAPI(title="anvil", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(auth.router)


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in auth.OPEN:
        user = auth.current_user(request)
        if user is None:
            return JSONResponse({"detail": "Log in first."}, status_code=401)
        request.state.user = user
    return await call_next(request)


def hf_token(request: Request) -> str | None:
    """The logged-in account's Hugging Face token, or the one for everyone in .env."""
    return request.state.user.hf_token or config.settings.hf_token


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


# ---------------------------------------------------------------- the account

def hf_status(user: accounts.User) -> dict:
    if user.hf_token:
        return {"connected": True, "user": user.hf_user, "source": "account"}
    if config.settings.hf_token:
        return {"connected": True, "user": None, "source": "env"}
    return {"connected": False, "user": None, "source": None}


def hub_whoami(token: str) -> dict:
    from huggingface_hub import HfApi

    return HfApi().whoami(token=token)


@app.get("/api/account")
def get_account(request: Request) -> dict:
    user = request.state.user
    return {"username": user.username, "hf": hf_status(user)}


class PasswordChange(BaseModel):
    current: str
    new: str


@app.put("/api/account/password")
def change_password(body: PasswordChange, request: Request, response: Response) -> dict:
    user = request.state.user
    if not accounts.check_password(body.current, user.password_hash):
        raise HTTPException(400, "Your current password isn't right.")
    if len(body.new) < 8:
        raise HTTPException(400, "Use at least 8 characters.")
    user = accounts.set_password(user.id, body.new)
    auth.set_cookie(response, user)  # the old cookie stopped working with the old password
    return {"ok": True}


class HfToken(BaseModel):
    token: str


@app.put("/api/account/hf-token")
def save_hf_token(body: HfToken, request: Request) -> dict:
    from huggingface_hub.errors import HfHubHTTPError

    token = body.token.strip()
    if not token.startswith("hf_"):
        raise HTTPException(400, "Hugging Face tokens start with hf_.")
    try:
        name = hub_whoami(token).get("name")
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise HTTPException(400, "Hugging Face didn't accept that token. Check it, or make a new one.") from exc
        raise HTTPException(502, "Couldn't reach Hugging Face to check the token.") from exc
    except Exception as exc:
        raise HTTPException(502, "Couldn't reach Hugging Face to check the token.") from exc
    user = accounts.set_hf_token(request.state.user.id, token, name)
    return {"hf": hf_status(user)}


@app.delete("/api/account/hf-token")
def remove_hf_token(request: Request) -> dict:
    return {"hf": hf_status(accounts.set_hf_token(request.state.user.id, None, None))}


# ---------------------------------------------------------------- GPUs and models

@app.get("/api/gpus")
def get_gpus() -> dict:
    return gpus.list_gpus()


@app.get("/api/hub/search")
def hub_search(request: Request, q: str = "", limit: int = 12) -> dict:
    try:
        return {"models": models_check.search(q.strip(), hf_token(request), max(1, min(limit, 50)))}
    except Exception as exc:  # offline, rate-limited, or the Hub is down
        raise HTTPException(502, f"Couldn't reach Hugging Face: {exc}") from exc


@app.get("/api/hub/check")
def hub_check(request: Request, repo: str) -> dict:
    if not REPO_ID.match(repo):
        raise HTTPException(400, "A model id looks like org/name.")
    try:
        return models_check.check_repo(repo, hf_token(request)).to_dict()
    except Exception as exc:
        raise HTTPException(502, f"Couldn't reach Hugging Face: {exc}") from exc


@app.get("/api/folders")
def get_folders(path: str | None = None) -> dict:
    try:
        return folders.list_dirs(path)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


class FolderCheck(BaseModel):
    path: str


@app.post("/api/folders/check")
def check_folder(body: FolderCheck) -> dict:
    try:
        path = folders.inside_roots(body.path)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not path.is_dir():
        raise HTTPException(404, f"{body.path} isn't a folder.")
    return models_check.check_folder(path).to_dict()


@app.get("/api/{rest:path}")
def unknown_api(rest: str):
    raise HTTPException(404, f"No such API route: /api/{rest}")


# The built screen. Paths that aren't files get index.html, so the screen handles its own pages.
@app.get("/{rest:path}")
def screen(rest: str):
    dist = config.settings.web_dist
    target = (dist / rest).resolve()
    if rest and target.is_file() and dist.resolve() in target.parents:
        return FileResponse(target)
    if (dist / "index.html").is_file():
        return FileResponse(dist / "index.html")
    return JSONResponse({"detail": "The screen isn't built. Use docker/dev.sh, or run `npm --prefix web run build`."},
                        status_code=404)
