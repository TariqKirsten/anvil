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

from . import accounts, auth, config, datasets, folders, gpus, models_check, runs

REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts.ensure_default_user()
    config.settings.uploads.mkdir(parents=True, exist_ok=True)
    config.settings.prepared.mkdir(parents=True, exist_ok=True)
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


# ---------------------------------------------------------------- datasets

def _hub_failure(exc: Exception) -> HTTPException:
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if isinstance(exc, GatedRepoError):
        return HTTPException(403, "This dataset is gated. Accept its terms on its Hub page, then add your Hugging "
                                  "Face token in Settings.")
    if isinstance(exc, RepositoryNotFoundError):
        return HTTPException(404, "There's no such dataset on the Hub, or it's private and your token can't see it.")
    return HTTPException(502, f"Couldn't reach Hugging Face: {exc}")


def _data_failure(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(403, str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    return _hub_failure(exc)


@app.get("/api/datasets/browse")
def browse_datasets(path: str | None = None) -> dict:
    try:
        return folders.list_dirs(path, config.settings.all_dataset_roots, datasets.EXTENSIONS)
    except (PermissionError, FileNotFoundError) as exc:
        raise _data_failure(exc) from exc


UPLOAD_NAME = re.compile(r"[^\w.-]+")


@app.put("/api/datasets/upload")
async def upload_dataset(request: Request, name: str) -> dict:
    """The file arrives as the request body, written to disk as it streams in."""
    clean = UPLOAD_NAME.sub("_", name.rsplit("/", 1)[-1]).strip("._") or "dataset"
    stem, dot, suffix = clean.rpartition(".")
    if not dot or f".{suffix.lower()}" not in datasets.EXTENSIONS:
        raise HTTPException(400, "Upload a .parquet, .jsonl, .json or .csv file.")
    folder = config.settings.uploads
    folder.mkdir(parents=True, exist_ok=True)
    target, n = folder / clean, 1
    while target.exists():
        target, n = folder / f"{stem}-{n}.{suffix}", n + 1
    partial = target.with_name(target.name + ".part")
    try:
        with partial.open("wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        partial.rename(target)
    finally:
        partial.unlink(missing_ok=True)
    return {"path": str(target), "name": target.name, "size": target.stat().st_size}


class InspectBody(BaseModel):
    source: dict


@app.post("/api/datasets/inspect")
def inspect_dataset(body: InspectBody, request: Request) -> dict:
    try:
        return datasets.inspect(body.source, hf_token(request))
    except Exception as exc:
        raise _data_failure(exc) from exc


class PrepareBody(BaseModel):
    train: dict
    validation: dict | None = None
    test: dict | None = None
    dedupe: bool = True
    near_copies: bool = True
    val_percent: float = 5
    test_percent: float = 5
    keep_together: str = "input"
    seed: int = 3407


@app.post("/api/datasets/prepare")
def prepare_dataset(body: PrepareBody, request: Request, run: bool = True) -> dict:
    """Start preparing (or find the earlier result). With run=false it only looks for an earlier result."""
    if not (0 <= body.val_percent <= 40 and 0 <= body.test_percent <= 40):
        raise HTTPException(400, "Use 0 to 40% for the validation and test splits.")
    if body.keep_together not in ("input", "source", "nothing"):
        raise HTTPException(400, "Keep rows together by input, source or nothing.")
    try:
        job = datasets.start(body.model_dump(), hf_token(request), run=run)
    except Exception as exc:
        raise _data_failure(exc) from exc
    return {"job": job.public() if job else None}


@app.get("/api/datasets/jobs/{job_id}")
def dataset_job(job_id: str) -> dict:
    job = datasets.job(job_id)
    if job is None:
        raise HTTPException(404, "That job is gone. anvil may have restarted; prepare the data again.")
    return {"job": job.public()}


@app.get("/api/hub/datasets/search")
def hub_dataset_search(request: Request, q: str = "") -> dict:
    try:
        return {"datasets": datasets.search_hub(q.strip(), hf_token(request))}
    except Exception as exc:
        raise _hub_failure(exc) from exc


@app.get("/api/hub/datasets/configs")
def hub_dataset_configs(request: Request, repo: str) -> dict:
    if not REPO_ID.match(repo):
        raise HTTPException(400, "A dataset id looks like org/name.")
    try:
        return {"configs": datasets.hub_configs(repo, hf_token(request))}
    except Exception as exc:
        raise _hub_failure(exc) from exc


@app.get("/api/hub/datasets/splits")
def hub_dataset_splits(request: Request, repo: str, config: str | None = None) -> dict:
    if not REPO_ID.match(repo):
        raise HTTPException(400, "A dataset id looks like org/name.")
    try:
        return {"splits": datasets.hub_splits(repo, config, hf_token(request))}
    except Exception as exc:
        raise _hub_failure(exc) from exc


# ---------------------------------------------------------------- runs

class RunBody(BaseModel):
    form: dict
    prepare: dict


@app.post("/api/runs")
def start_run(body: RunBody, request: Request) -> dict:
    try:
        return {"id": runs.create(body.form, body.prepare, hf_token(request))}
    except (KeyError, TypeError) as exc:
        raise HTTPException(400, f"The run settings are incomplete ({exc}). Reload the page and try again.") from exc
    except Exception as exc:
        raise _data_failure(exc) from exc


@app.get("/api/runs")
def list_runs() -> dict:
    return {"runs": runs.list_runs()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return runs.detail(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    try:
        return {"freed": runs.delete(run_id)}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


class ExportBody(BaseModel):
    version: str
    quants: list[str]


@app.post("/api/runs/{run_id}/exports")
def start_export(run_id: str, body: ExportBody) -> dict:
    try:
        runs.start_export(run_id, body.version, body.quants)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@app.get("/api/runs/{run_id}/exports/{version}/{name}")
def download_export(run_id: str, version: str, name: str):
    try:
        return FileResponse(runs.export_file(run_id, version, name), filename=name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    try:
        runs.stop(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, ProcessLookupError) as exc:
        raise HTTPException(409, str(exc) or "This run isn't running.") from exc
    return {"ok": True}


class ResumeBody(BaseModel):
    patience: int | None = None  # more checks without a gain to wait; 0 trains to the last step


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str, request: Request, body: ResumeBody | None = None) -> dict:
    try:
        runs.resume(run_id, hf_token(request), body.patience if body else None)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


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
