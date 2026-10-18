"""Settings, read from the environment. compose passes .env through (see .env.example for what each one means)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _paths(value: str) -> tuple[Path, ...]:
    return tuple(Path(p).expanduser() for p in value.split(":") if p.strip())


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # The admin account's first password. It's only read when the account is created, or reset (below).
    password: str = field(default_factory=lambda: _env("ANVIL_PASSWORD"))
    reset_password: bool = field(default_factory=lambda: _env("ANVIL_RESET_PASSWORD") == "1")
    data: Path = field(default_factory=lambda: Path(_env("ANVIL_DATA", "~/.local/share/anvil")).expanduser())
    model_roots: tuple[Path, ...] = field(default_factory=lambda: _paths(_env("ANVIL_MODEL_ROOTS")))
    hf_token: str | None = field(default_factory=lambda: _env("HF_TOKEN") or None)
    # The built screen (npm run build). In development Vite serves it instead.
    web_dist: Path = Path(__file__).resolve().parents[2] / "web" / "dist"


settings = Settings()
