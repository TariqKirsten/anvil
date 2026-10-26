"""Talking to an OpenAI-compatible endpoint: the models it serves, and chat completions, with tools or without.

Any server that speaks OpenAI's API will do: vLLM, llama.cpp's llama-server, radiance, Ollama, LM Studio, OpenAI
itself or OpenRouter. A pipeline's Model step (anvil.pipelines) holds the endpoint's base URL, the model and how to ask
it. Its API key stays on the server and never goes back to the screen.

anvil runs in a container, where localhost is the container itself: a server on this machine is host.docker.internal.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

ENDPOINT_PATHS = ("/chat/completions", "/completions", "/models")
THOUGHTS = re.compile(r"<think>.*?</think>", re.S)
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class EndpointError(Exception):
    """A request the endpoint refused or couldn't answer, with a reason written to be shown as it is. fatal: no
    request to this endpoint will work until something changes (the address, the key or the model)."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


@dataclass
class Reply:
    text: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cut: bool = False  # the reply stopped at max tokens


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON, as the model wrote it


@dataclass
class Answer:
    """A reply that may call tools (OpenAI's function calling)."""
    text: str
    calls: list[ToolCall]
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cut: bool = False


def base_url(url: str) -> str:
    """The base URL the way OpenAI's clients take it, such as http://host:8000/v1: no endpoint path, no trailing /."""
    url = url.strip().rstrip("/")
    for path in ENDPOINT_PATHS:
        if url.endswith(path):
            url = url[: -len(path)].rstrip("/")
    return url


def check_url(url: str) -> str | None:
    """Why a base URL can't work, or None."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "An endpoint's address starts with http:// or https://, such as http://host.docker.internal:8000/v1."
    return None


def _headers(key: str | None) -> dict:
    return {"Authorization": f"Bearer {key}"} if key else {}


def _detail(response: httpx.Response) -> str:
    """What the endpoint said was wrong, from OpenAI's {"error": {"message": ...}} or whatever else it sent."""
    try:
        data = response.json()
    except ValueError:
        return " ".join(response.text.split())[:300]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        for key in ("error", "detail", "message"):
            if isinstance(data.get(key), str):
                return data[key][:300]
    return str(data)[:300]


def _refused(response: httpx.Response, url: str) -> EndpointError:
    detail = _detail(response)
    said = f" It said: {detail}" if detail else ""
    if response.status_code in (401, 403):
        return EndpointError(f"The endpoint didn't accept the API key ({response.status_code}).{said}", fatal=True)
    if response.status_code == 404:
        return EndpointError(f"Nothing answers at {url} (404). The address may need /v1 at the end, or the model "
                             f"name may be wrong.{said}", fatal=True)
    return EndpointError(f"The endpoint answered {response.status_code}.{said}")


def _unreachable(url: str, error: Exception) -> EndpointError:
    host = urlparse(url).hostname or ""
    hint = ""
    if host in ("localhost", "127.0.0.1", "::1"):
        hint = (" Inside anvil's container, localhost is the container itself: use host.docker.internal for this "
                "machine.")
    if isinstance(error, httpx.TimeoutException):
        return EndpointError(f"{url} didn't answer in time.{hint}")
    return EndpointError(f"Couldn't reach {url}: {error.__class__.__name__}.{hint}")


def list_models(url: str, key: str | None, timeout: float = 15) -> tuple[str, list[str]]:
    """(base URL, the ids of the models it serves). A base URL without /v1 that only answers with it is given back
    with /v1, so the screen can correct what was typed."""
    base = base_url(url)
    problem = check_url(base)
    if problem:
        raise EndpointError(problem, fatal=True)
    tries = [base] if base.endswith("/v1") else [base, base + "/v1"]
    for i, candidate in enumerate(tries):
        try:
            response = httpx.get(f"{candidate}/models", headers=_headers(key), timeout=timeout)
        except httpx.HTTPError as error:
            raise _unreachable(candidate, error) from error
        if response.status_code == 404 and i + 1 < len(tries):
            continue
        if response.status_code != 200:
            raise _refused(response, f"{candidate}/models")
        try:
            data = response.json()
            models = [str(m["id"]) for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        except (ValueError, AttributeError, TypeError):
            raise EndpointError(f"{candidate}/models didn't answer like an OpenAI-compatible endpoint.") from None
        return candidate, sorted(models)
    raise EndpointError(f"Nothing answers at {base}/models.")  # unreachable: the loop returns or raises


def answer_text(content: str | None) -> str:
    """A reply without the model's thinking: <think> blocks, or everything up to a lone </think> when the server put
    the opening tag in the prompt."""
    text = THOUGHTS.sub("", content or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


class Client:
    """One endpoint and model, shared by every thread of a pipeline run (httpx's client is safe across threads)."""

    def __init__(self, url: str, key: str | None, model: str, *, temperature: float | None = None,
                 max_tokens: int | None = None, thinking: str = "default", extra: dict | None = None,
                 timeout: float = 300, retries: int = 3):
        self.url = base_url(url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.extra = dict(extra or {})
        self.retries = retries
        self.http = httpx.Client(headers=_headers(key), timeout=httpx.Timeout(timeout, connect=15))

    def close(self) -> None:
        self.http.close()

    def payload(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> dict:
        body: dict = {"model": self.model, "messages": messages}
        temperature = self.temperature if temperature is None else temperature
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens or self.max_tokens:
            body["max_tokens"] = max_tokens or self.max_tokens
        if self.thinking in ("on", "off"):  # Qwen's, Gemma's and DeepSeek's chat templates read this
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking == "on"}
        body.update(self.extra)
        return body

    def _complete(self, body: dict) -> tuple[dict, dict, float]:
        """POST a chat completion: (the first choice, the usage, seconds). Network trouble, rate limits and server
        errors are tried again, waiting 1, 2, 4… s; anything else the endpoint refuses raises EndpointError at once."""
        url = f"{self.url}/chat/completions"
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            last = attempt == self.retries
            try:
                response = self.http.post(url, json=body)
            except httpx.HTTPError as error:
                if last:
                    raise _unreachable(url, error) from error
                time.sleep(2 ** attempt)
                continue
            if response.status_code in RETRY_STATUS and not last:
                wait = response.headers.get("retry-after", "")
                time.sleep(min(60.0, float(wait)) if wait.replace(".", "", 1).isdigit() else 2 ** attempt)
                continue
            if response.status_code != 200:
                raise _refused(response, url)
            try:
                data = response.json()
                choice = data["choices"][0]
                if not isinstance(choice["message"], dict):
                    raise TypeError
            except (ValueError, KeyError, IndexError, TypeError):
                raise EndpointError(f"{url} didn't answer like an OpenAI-compatible endpoint.") from None
            return choice, data.get("usage") or {}, time.monotonic() - started
        raise EndpointError(f"{url} kept failing.")  # unreachable: the last attempt returns or raises

    def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> Reply:
        """One chat completion, its text without the model's thinking."""
        choice, usage, seconds = self._complete(self.payload(messages, temperature, max_tokens))
        cut = choice.get("finish_reason") == "length"
        text = answer_text(choice["message"].get("content"))
        if not text and cut:
            raise EndpointError("The reply hit max tokens before it said anything (the model may have spent them "
                                "thinking). Raise Max tokens, or turn thinking off.")
        return Reply(text, seconds, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), cut)

    def answer(self, messages: list[dict], tools: list[dict]) -> Answer:
        """One chat completion the model may answer by calling tools: its text, without thinking, and the calls."""
        choice, usage, seconds = self._complete({**self.payload(messages), "tools": tools})
        calls = []
        for i, raw in enumerate(choice["message"].get("tool_calls") or []):
            function = raw.get("function") if isinstance(raw, dict) else None
            if not isinstance(function, dict) or not function.get("name"):
                continue
            arguments = function.get("arguments")
            calls.append(ToolCall(str(raw.get("id") or f"call_{i}"), str(function["name"]),
                                  arguments if isinstance(arguments, str) else json.dumps(arguments or {})))
        return Answer(answer_text(choice["message"].get("content")), calls, seconds,
                      int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
                      choice.get("finish_reason") == "length")
