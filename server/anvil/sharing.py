"""The GPUs shared with another program on this machine, such as a model server that answers pipelines and the
assistant. Before training starts, anvil can ask it to let go of the cards, by calling an address you give in
Settings (a POST, such as a server's unload endpoint).

While a training run uses the cards, anvil sends no requests to models at that server (the same host and port), since
answering one would load a model back onto them: a pipeline run there doesn't start, and the assistant waits.

On disk: ANVIL_DATA/sharing.json ({"url"}) and ANVIL_DATA/sharing-secret.json ({"key"}, 0600). Without a key of its
own, a call to the assistant's server uses the assistant's key.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from . import config, llm
from .pipelines import _read, _write

FREE_TIMEOUT_S = 180  # a server stopping a model may take a minute


def _files():
    return config.settings.data / "sharing.json", config.settings.data / "sharing-secret.json"


def url() -> str:
    return str((_read(_files()[0], {}) or {}).get("url") or "")


def key() -> str | None:
    return (_read(_files()[1], {}) or {}).get("key") or None


def settings() -> dict:
    return {"url": url(), "has_key": bool(key())}


def save(address: str) -> dict:
    """Save the address to call before training ("" for none). ValueError says what's wrong with it."""
    address = address.strip()
    if address:
        problem = llm.check_url(address)
        if problem:
            raise ValueError(problem.replace("An endpoint's address", "The address"))
    config.settings.data.mkdir(parents=True, exist_ok=True)
    _write(_files()[0], {"url": address})
    return settings()


def set_key(value: str | None) -> None:
    config.settings.data.mkdir(parents=True, exist_ok=True)
    _write(_files()[1], {"key": value} if value else {}, private=True)


def server(address: str) -> str:
    """scheme://host:port of an address, so an endpoint and the address to call can be told to be the same server."""
    parsed = urlparse(address.strip())
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.scheme}://{(parsed.hostname or '').lower()}:{port}"


def _key_for_call(address: str) -> str | None:
    from . import assistant  # assistant builds on pipelines, as this module does

    if key():
        return key()
    mine = assistant.settings()["base_url"]
    return assistant.key() if server(mine) == server(address) else None


def free_cards() -> str | None:
    """Call the address, if there is one: a line on how it went. Nothing answering means nothing holds the cards, so
    that's fine too; an answer that isn't a success raises ValueError, so training doesn't start on cards still held."""
    address = url()
    if not address:
        return None
    secret = _key_for_call(address)
    try:
        response = httpx.post(address, headers={"Authorization": f"Bearer {secret}"} if secret else {},
                              timeout=httpx.Timeout(FREE_TIMEOUT_S, connect=10))
    except httpx.HTTPError as error:
        return f"Nothing answered at {address} ({error.__class__.__name__}), so nothing there holds the GPUs."
    if response.status_code >= 300:
        detail = " ".join(response.text.split())[:200]
        raise ValueError(f"Couldn't free the GPUs: {address} answered {response.status_code}"
                         + (f" ({detail})" if detail else "") + ". Free them yourself, or change the address in "
                         "Settings.")
    return f"Asked {address} to let go of the GPUs."


def training() -> str | None:
    """The anvil run training on the GPUs now, if any."""
    from . import runs  # runs calls free_cards before it trains

    busy = runs.busy_gpus()
    return next(iter(busy.values()), None)


def held(endpoint: str) -> str | None:
    """The run training now, when endpoint is at the server whose GPUs training uses (the address to call): a request
    there would load a model back onto them."""
    address = url()
    if not address or not endpoint or server(endpoint) != server(address):
        return None
    return training()
