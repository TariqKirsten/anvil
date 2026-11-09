"""The anvil server: the API under /api, and the built screen at every other path.

Run it with docker/dev.sh (development) or docker/app.sh (the finished app), inside the container.
"""

from __future__ import annotations

import re
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from . import (accounts, assistant, auth, config, datasets, folders, gpus, llm, models_check, pipelines, runs, sharing,
               templates)

REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts.ensure_default_user()
    config.settings.uploads.mkdir(parents=True, exist_ok=True)
    config.settings.prepared.mkdir(parents=True, exist_ok=True)
    config.settings.generated.mkdir(parents=True, exist_ok=True)
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
    val_percent: float = 5
    test_percent: float = 5
    keep_together: str = "input"  # input, nothing, or column:<name>
    balance: bool = True  # decision data: each answer gets its share of validation and test
    seed: int = 3407


@app.post("/api/datasets/prepare")
def prepare_dataset(body: PrepareBody, request: Request, run: bool = True) -> dict:
    """Start preparing (or find the earlier result). With run=false it only looks for an earlier result."""
    if not (0 <= body.val_percent <= 40 and 0 <= body.test_percent <= 40):
        raise HTTPException(400, "Use 0 to 40% for the validation and test splits.")
    together = body.keep_together
    if together not in ("input", "source", "nothing") and not (together.startswith("column:") and together[7:].strip()):
        raise HTTPException(400, "Keep rows together by input, a column (column:<name>) or nothing.")
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


class UploadBody(BaseModel):
    version: str
    weights: str | None = None  # "merged" (the whole model) or "adapter" (LoRA's alone); None: GGUF files only
    gguf: list[str] = []  # GGUF files from the version's export
    repo: str | None = None  # where the weights go
    gguf_repo: str | None = None  # where the GGUF files go (the same repo is fine)
    private: bool = True


@app.post("/api/runs/{run_id}/uploads")
def start_upload(run_id: str, body: UploadBody, request: Request) -> dict:
    try:
        return {"id": runs.start_upload(run_id, body.model_dump(), hf_token(request))}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/runs/{run_id}/uploads/{upload_id}/cancel")
def cancel_upload(run_id: str, upload_id: str) -> dict:
    try:
        runs.cancel_upload(run_id, upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


_accounts: dict[str, tuple[float, dict]] = {}  # token -> (when, what whoami said), so the screen can ask often


def hub_account(token: str) -> dict:
    """Whose a token is, and where it can upload: the account's own namespace and the organisations it may write to.
    write: whether it can write at all (a read token can't)."""
    cached = _accounts.get(token)
    if cached and time.time() - cached[0] < 300:  # Hugging Face limits how often whoami may be asked
        return cached[1]
    info = hub_whoami(token)
    access = (info.get("auth") or {}).get("accessToken") or {}
    name = info.get("name")
    orgs = [o["name"] for o in info.get("orgs") or [] if o.get("name") and o.get("roleInOrg") != "read"]
    if access.get("role") == "fineGrained":
        scoped = (access.get("fineGrained") or {}).get("scoped") or []
        places = [s["entity"]["name"] for s in scoped if "repo.write" in (s.get("permissions") or [])
                  and (s.get("entity") or {}).get("name")]
        namespaces = [n for n in [name, *orgs] if n in places]
    elif access.get("role") == "read":
        namespaces = []
    else:  # a write token (or one from before roles): the account's own, and its organisations'
        namespaces = [name, *orgs]
    found = {"name": name, "namespaces": namespaces, "write": bool(namespaces)}
    _accounts[token] = (time.time(), found)
    return found


@app.get("/api/hub/me")
def hub_me(request: Request) -> dict:
    token = hf_token(request)
    if not token:
        return {"connected": False, "name": None, "namespaces": [], "write": False}
    try:
        return {"connected": True, **hub_account(token)}
    except Exception as exc:
        raise HTTPException(502, "Couldn't reach Hugging Face to see whose the token is.") from exc


@app.get("/api/hub/repo")
def hub_repo(request: Request, repo: str) -> dict:
    """Whether a repo exists (that the token can see), and if so whether it's private."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    if not REPO_ID.match(repo):
        raise HTTPException(400, "A repository looks like your-name/model-name.")
    try:
        info = HfApi().repo_info(repo, token=hf_token(request))
    except RepositoryNotFoundError:
        return {"exists": False, "private": None}
    except Exception as exc:
        raise HTTPException(502, f"Couldn't reach Hugging Face: {exc}") from exc
    return {"exists": True, "private": bool(info.private)}


class StopBody(BaseModel):
    now: bool = False  # Stop now: end the trainer at once, not after this step (losing the steps since its checkpoint)


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str, body: StopBody | None = None) -> dict:
    try:
        runs.stop(run_id, now=bool(body and body.now))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, ProcessLookupError) as exc:
        raise HTTPException(409, str(exc) or "This run isn't running.") from exc
    return {"ok": True}


class SuperviseBody(BaseModel):
    on: bool  # restart the trainer after a crash or a freeze, or leave the run to fail or wait for Stop


@app.post("/api/runs/{run_id}/supervise")
def supervise_run(run_id: str, body: SuperviseBody) -> dict:
    try:
        runs.supervise(run_id, body.on)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True}


class ResumeBody(BaseModel):
    patience: int | None = None  # more checks without a gain to wait; 0 trains to the last step
    epochs: int | None = None  # passes over the training data in all, from the start; lifts a Max steps limit


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str, request: Request, body: ResumeBody | None = None) -> dict:
    try:
        runs.resume(run_id, hf_token(request), body.patience if body else None, body.epochs if body else None)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


# ---------------------------------------------------------------- pipelines: making datasets with a model

def _pipeline_failure(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, str(exc))
    if isinstance(exc, llm.EndpointError):  # the endpoint's trouble, not anvil's
        return HTTPException(502, str(exc))
    return HTTPException(409, str(exc))


@app.get("/api/pipeline-steps")
def pipeline_steps() -> dict:
    """Every step's settings with their defaults and the tests a Filter can use, so the screen makes a new step the
    way the server would fill it in."""
    return {"steps": pipelines.STEPS, "filter_ops": pipelines.FILTER_OPS, "leave_out": pipelines.LEAVE_OUT}


class PipelineNew(BaseModel):
    name: str = ""
    template: str | None = None  # start from one of anvil.templates


@app.get("/api/pipeline-templates")
def pipeline_templates() -> dict:
    return {"templates": templates.public()}


@app.get("/api/pipelines")
def list_pipelines() -> dict:
    return {"pipelines": pipelines.list_pipelines()}


@app.post("/api/pipelines")
def create_pipeline(body: PipelineNew) -> dict:
    try:
        return {"pipeline": pipelines.detail(pipelines.create(body.name, body.template)["id"])}
    except FileNotFoundError as exc:
        raise _pipeline_failure(exc) from exc


@app.get("/api/pipelines/{pipeline_id}")
def get_pipeline(pipeline_id: str) -> dict:
    try:
        return {"pipeline": pipelines.detail(pipeline_id)}
    except FileNotFoundError as exc:
        raise _pipeline_failure(exc) from exc


class PipelineBody(BaseModel):
    name: str | None = None
    nodes: list[dict] = []
    edges: list[dict] = []


def _not_while_assistant(pipeline_id: str) -> None:
    if assistant.working(pipeline_id):
        raise HTTPException(409, "The assistant is working on this pipeline: wait for it to finish, or stop it.")


@app.put("/api/pipelines/{pipeline_id}")
def save_pipeline(pipeline_id: str, body: PipelineBody) -> dict:
    """Save the graph as the screen has it (half drawn is fine). The answer says what still stops it from running."""
    _not_while_assistant(pipeline_id)
    try:
        data = pipelines.save(pipeline_id, body.model_dump(exclude_none=True))
    except FileNotFoundError as exc:
        raise _pipeline_failure(exc) from exc
    return {"updated": data["updated"], "problems": pipelines.check(data)}


@app.delete("/api/pipelines/{pipeline_id}")
def delete_pipeline(pipeline_id: str) -> dict:
    _not_while_assistant(pipeline_id)
    try:
        pipelines.delete(pipeline_id)
    except (FileNotFoundError, ValueError) as exc:
        raise _pipeline_failure(exc) from exc
    return {"ok": True}


class KeyBody(BaseModel):
    key: str


@app.put("/api/pipelines/{pipeline_id}/keys/{node}")
def save_pipeline_key(pipeline_id: str, node: str, body: KeyBody) -> dict:
    if not body.key.strip():
        raise HTTPException(400, "Paste the key first.")
    try:
        pipelines.set_key(pipeline_id, node, body.key.strip())
    except FileNotFoundError as exc:
        raise _pipeline_failure(exc) from exc
    return {"ok": True}


@app.delete("/api/pipelines/{pipeline_id}/keys/{node}")
def remove_pipeline_key(pipeline_id: str, node: str) -> dict:
    try:
        pipelines.set_key(pipeline_id, node, None)
    except FileNotFoundError as exc:
        raise _pipeline_failure(exc) from exc
    return {"ok": True}


class ModelsBody(BaseModel):
    base_url: str


@app.post("/api/pipelines/{pipeline_id}/nodes/{node}/models")
def endpoint_models(pipeline_id: str, node: str, body: ModelsBody) -> dict:
    """The models a Model step's endpoint serves, asked with the step's key."""
    try:
        return pipelines.endpoint_models(pipeline_id, node, body.base_url)
    except (FileNotFoundError, llm.EndpointError) as exc:
        raise _pipeline_failure(exc) from exc


class TryBody(BaseModel):
    message: str
    config: dict | None = None  # the Model step's settings as the screen has them, saved or not


@app.post("/api/pipelines/{pipeline_id}/nodes/{node}/try")
def try_pipeline_model(pipeline_id: str, node: str, body: TryBody) -> dict:
    if not body.message.strip():
        raise HTTPException(400, "Write a message to send.")
    try:
        return pipelines.try_model(pipeline_id, node, body.config, body.message)
    except (FileNotFoundError, llm.EndpointError) as exc:
        raise _pipeline_failure(exc) from exc


class PipelineStart(BaseModel):
    preview: bool = False


@app.post("/api/pipelines/{pipeline_id}/runs")
def start_pipeline(pipeline_id: str, body: PipelineStart, request: Request) -> dict:
    _not_while_assistant(pipeline_id)
    try:
        return {"id": pipelines.start(pipeline_id, body.preview, hf_token(request))}
    except (FileNotFoundError, ValueError) as exc:
        raise _pipeline_failure(exc) from exc


@app.get("/api/pipelines/{pipeline_id}/runs/{run_id}")
def pipeline_run(pipeline_id: str, run_id: str) -> dict:
    try:
        return {"run": pipelines.run_status(pipeline_id, run_id)}
    except FileNotFoundError as exc:
        raise _pipeline_failure(exc) from exc


@app.get("/api/pipelines/{pipeline_id}/runs/{run_id}/rows")
def pipeline_rows(pipeline_id: str, run_id: str, which: str, offset: int = 0, limit: int = 50) -> dict:
    """A Save's rows (which = its step's id), the rows the steps dropped ("dropped"), or a preview's ("preview")."""
    try:
        return pipelines.rows(pipeline_id, run_id, which, max(0, offset), max(1, min(limit, 200)))
    except FileNotFoundError as exc:
        raise _pipeline_failure(exc) from exc


@app.get("/api/pipelines/{pipeline_id}/runs/{run_id}/log")
def pipeline_log(pipeline_id: str, run_id: str) -> dict:
    try:
        return {"log": pipelines.log(pipeline_id, run_id)}
    except FileNotFoundError as exc:
        raise _pipeline_failure(exc) from exc


@app.post("/api/pipelines/{pipeline_id}/runs/{run_id}/stop")
def stop_pipeline(pipeline_id: str, run_id: str) -> dict:
    try:
        pipelines.stop(pipeline_id, run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise _pipeline_failure(exc) from exc
    return {"ok": True}


@app.post("/api/pipelines/{pipeline_id}/runs/{run_id}/resume")
def resume_pipeline(pipeline_id: str, run_id: str, request: Request) -> dict:
    try:
        pipelines.resume(pipeline_id, run_id, hf_token(request))
    except (FileNotFoundError, ValueError) as exc:
        raise _pipeline_failure(exc) from exc
    return {"ok": True}


@app.delete("/api/pipelines/{pipeline_id}/runs/{run_id}")
def delete_pipeline_run(pipeline_id: str, run_id: str) -> dict:
    try:
        pipelines.delete_run(pipeline_id, run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise _pipeline_failure(exc) from exc
    return {"ok": True}


# ---------------------------------------------------------------- the pipeline assistant

class AssistantSettings(BaseModel):
    base_url: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    thinking: str | None = None
    reasoning_effort: str | None = None
    context: int | None = None
    timeout: int | None = None


@app.get("/api/assistant")
def assistant_settings() -> dict:
    return {"settings": assistant.settings()}


@app.put("/api/assistant")
def save_assistant_settings(body: AssistantSettings) -> dict:
    given = body.model_dump(exclude_unset=True)
    try:
        return {"settings": assistant.save_settings(given)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/assistant/key")
def save_assistant_key(body: KeyBody) -> dict:
    if not body.key.strip():
        raise HTTPException(400, "Paste the key first.")
    assistant.set_key(body.key.strip())
    return {"settings": assistant.settings()}


@app.delete("/api/assistant/key")
def remove_assistant_key() -> dict:
    assistant.set_key(None)
    return {"settings": assistant.settings()}


class AssistantModels(BaseModel):
    base_url: str | None = None


@app.post("/api/assistant/models")
def assistant_models(body: AssistantModels) -> dict:
    """The models an address serves, asked with the assistant's key."""
    try:
        return assistant.models(body.base_url)
    except llm.EndpointError as exc:
        raise _pipeline_failure(exc) from exc


@app.get("/api/pipelines/{pipeline_id}/assistant")
def pipeline_assistant(pipeline_id: str) -> dict:
    try:
        return assistant.conversation(pipeline_id)
    except FileNotFoundError as exc:
        raise _pipeline_failure(exc) from exc


class AssistantMessage(BaseModel):
    message: str


@app.post("/api/pipelines/{pipeline_id}/assistant")
def message_assistant(pipeline_id: str, body: AssistantMessage, request: Request) -> dict:
    """Send the assistant a message: it works on it in a process of its own; the conversation says how it goes."""
    try:
        return assistant.start_turn(pipeline_id, body.message, hf_token(request))
    except (FileNotFoundError, ValueError) as exc:
        raise _pipeline_failure(exc) from exc


@app.post("/api/pipelines/{pipeline_id}/assistant/stop")
def stop_assistant(pipeline_id: str) -> dict:
    try:
        assistant.stop_turn(pipeline_id)
    except (FileNotFoundError, ValueError) as exc:
        raise _pipeline_failure(exc) from exc
    return {"ok": True}


@app.post("/api/pipelines/{pipeline_id}/assistant/go")
def assistant_ask_anyway(pipeline_id: str) -> dict:
    """The assistant waits while a run uses its address with another model: ask it anyway."""
    try:
        assistant.ask_anyway(pipeline_id)
    except (FileNotFoundError, ValueError) as exc:
        raise _pipeline_failure(exc) from exc
    return {"ok": True}


@app.post("/api/pipelines/{pipeline_id}/assistant/undo")
def undo_assistant(pipeline_id: str) -> dict:
    try:
        return assistant.undo(pipeline_id)
    except (FileNotFoundError, ValueError) as exc:
        raise _pipeline_failure(exc) from exc


@app.delete("/api/pipelines/{pipeline_id}/assistant")
def clear_assistant(pipeline_id: str) -> dict:
    """Start over: the conversation goes; the pipeline stays as it is."""
    try:
        assistant.clear(pipeline_id)
    except (FileNotFoundError, ValueError) as exc:
        raise _pipeline_failure(exc) from exc
    return {"ok": True}


# ---------------------------------------------------------------- GPUs shared with a model server

class SharingBody(BaseModel):
    url: str


@app.get("/api/sharing")
def sharing_settings() -> dict:
    return {"sharing": sharing.settings()}


@app.put("/api/sharing")
def save_sharing(body: SharingBody) -> dict:
    """The address anvil calls (POST) before training, so a model server on these GPUs lets go of them."""
    try:
        return {"sharing": sharing.save(body.url)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/sharing/key")
def save_sharing_key(body: KeyBody) -> dict:
    if not body.key.strip():
        raise HTTPException(400, "Paste the key first.")
    sharing.set_key(body.key.strip())
    return {"sharing": sharing.settings()}


@app.delete("/api/sharing/key")
def remove_sharing_key() -> dict:
    sharing.set_key(None)
    return {"sharing": sharing.settings()}


@app.post("/api/sharing/free")
def free_cards_now() -> dict:
    """Call the address now, as training would."""
    if not sharing.url():
        raise HTTPException(409, "Save an address first.")
    try:
        return {"note": sharing.free_cards()}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


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
