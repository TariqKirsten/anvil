"""Browsing for model folders and dataset files, limited to the folders anvil is told about.

anvil is reachable from the network, so the screen must not be able to look around the whole disk.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def inside_roots(path: str | Path, roots: tuple[Path, ...] | None = None) -> Path:
    """The resolved path, if it lies inside one of the roots (the model folders by default). PermissionError if not."""
    roots = config.settings.model_roots if roots is None else roots
    resolved = Path(path).expanduser().resolve()
    for root in roots:
        root = root.resolve()
        if resolved == root or root in resolved.parents:
            return resolved
    raise PermissionError(f"{path} is outside the folders anvil can see. Pick one of the folders below, or mount the "
                          "folder you want (see .env).")


def list_dirs(path: str | None, roots: tuple[Path, ...] | None = None, extensions: tuple[str, ...] = ()) -> dict:
    """The roots when no path is given, otherwise the folders inside path (plus files with the given extensions)."""
    roots = config.settings.model_roots if roots is None else roots
    names = [str(r) for r in roots if r.is_dir()]
    if not path:
        return {"path": None, "parent": None, "dirs": names, "files": [], "roots": names}
    here = inside_roots(path, roots)
    if not here.is_dir():
        raise FileNotFoundError(f"{path} isn't a folder.")
    children = sorted(here.iterdir(), key=lambda c: c.name.lower())
    dirs = [c.name for c in children if c.is_dir() and not c.name.startswith(".")]
    files = [{"name": c.name, "size": c.stat().st_size} for c in children
             if extensions and c.is_file() and c.suffix.lower() in extensions and not c.name.startswith(".")]
    at_root = any(here == Path(r).resolve() for r in names)
    return {"path": str(here), "parent": None if at_root else str(here.parent), "dirs": dirs[:500],
            "files": files[:500], "roots": names}
