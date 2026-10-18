"""Browsing for a local model folder, limited to the folders listed in ANVIL_MODEL_ROOTS.

anvil is reachable from the network, so the screen must not be able to look around the whole disk.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def inside_roots(path: str | Path) -> Path:
    """The resolved path, if it lies inside one of the model roots. Raises PermissionError otherwise."""
    resolved = Path(path).expanduser().resolve()
    for root in config.settings.model_roots:
        root = root.resolve()
        if resolved == root or root in resolved.parents:
            return resolved
    raise PermissionError(f"{path} is outside the folders anvil can see. Pick one of the model folders below, or "
                          "mount the folder you want (ANVIL_MODELS in .env).")


def list_dirs(path: str | None) -> dict:
    """The model roots when no path is given, otherwise the folders inside path and the way back up."""
    roots = [str(r) for r in config.settings.model_roots if r.is_dir()]
    if not path:
        return {"path": None, "parent": None, "dirs": roots, "roots": roots}
    here = inside_roots(path)
    if not here.is_dir():
        raise FileNotFoundError(f"{path} isn't a folder.")
    dirs = sorted(c.name for c in here.iterdir() if c.is_dir() and not c.name.startswith("."))
    at_root = any(here == Path(r).resolve() for r in roots)
    return {"path": str(here), "parent": None if at_root else str(here.parent), "dirs": dirs[:500], "roots": roots}
