"""Can this model be trained, and what is it?

The checks work from a model's list of files plus its small JSON files, so the same code checks a local folder and a
Hugging Face repo. A trainable model needs:
- config.json, which says what kind of model it is, in a form the installed transformers knows (or custom code)
- weights in safetensors (old .bin files load too, but they're slower and can run code)
- tokenizer files
GGUF files are for running a model, not training it, so a GGUF-only folder is refused. A LoRA adapter folder is
recognised and names the base model it needs.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "spiece.model")
PERMISSIVE = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0"}
# Items marked "info" are neutral notes: they never change the verdict.
STATUSES = ("ok", "info", "warn", "bad")


@dataclass
class Item:
    status: str
    label: str
    detail: str


@dataclass
class Report:
    verdict: str  # "ok", "warn" or "bad"
    message: str
    kind: str  # "model", "decision", "adapter" or "unknown"
    architecture: str | None = None
    params: int | None = None
    size: int = 0
    base_model: str | None = None
    items: list[Item] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def transformers_version() -> str:
    import transformers

    return transformers.__version__


def known_model_types() -> set[str]:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    return set(CONFIG_MAPPING_NAMES)


def _verdict(items: list[Item]) -> str:
    statuses = {i.status for i in items}
    return "bad" if "bad" in statuses else "warn" if "warn" in statuses else "ok"


def check(files: dict[str, int], read_json: Callable[[str], dict | None], params: int | None = None) -> Report:
    """files maps each top-level file name to its size in bytes; read_json(name) returns a parsed JSON file or None."""
    names = set(files)
    size = sum(files.values())
    safetensors = sorted(n for n in names if n.endswith(".safetensors"))
    ggufs = [n for n in names if n.endswith(".gguf")]
    bins = [n for n in names if n.startswith("pytorch_model") and n.endswith(".bin")]
    items: list[Item] = []

    if "adapter_config.json" in names:
        cfg = read_json("adapter_config.json") or {}
        base = cfg.get("base_model_name_or_path")
        weights = [n for n in ("adapter_model.safetensors", "adapter_model.bin") if n in names]
        items.append(Item("ok", "adapter_config.json", f"{cfg.get('peft_type', 'adapter')}, rank {cfg.get('r', '?')}"))
        items.append(Item("ok" if weights else "bad", "Adapter weights", gb(files[weights[0]]) if weights else "missing"))
        items.append(Item("ok" if base else "bad", "Base model", base or "not named in adapter_config.json"))
        if _verdict(items) == "bad":
            return Report("bad", "This adapter is incomplete.", "adapter", size=size, base_model=base, items=items)
        return Report("warn", "This is a LoRA adapter. Its base model gets loaded too.", "adapter",
                      params=params, size=size, base_model=base, items=items)

    config = read_json("config.json") if "config.json" in names else None
    if config is None:
        if ggufs:
            items += [Item("bad", "Weights", "only .gguf files"), Item("bad", "config.json", "missing")]
            return Report("bad", "GGUF is for running a model, not training it. Pick the original safetensors "
                                 "folder or repo instead.", "unknown", size=size, items=items)
        items.append(Item("bad", "config.json", "missing"))
        return Report("bad", "There's no config.json, so this isn't a model folder.", "unknown", size=size, items=items)

    architecture = (config.get("architectures") or [None])[0]
    model_type = config.get("model_type")
    items.append(Item("ok", "config.json", architecture or model_type or "no architecture named"))
    if config.get("auto_map") or any(n.endswith(".py") for n in names):
        items.append(Item("warn", "Custom code", "runs code that ships with the model (trust_remote_code), so only "
                                                 "use it if you trust the authors"))
    elif model_type in known_model_types():
        items.append(Item("ok", "Architecture", f"{model_type}, known to transformers {transformers_version()}"))
    else:
        items.append(Item("bad", "Architecture", f"transformers {transformers_version()} doesn't know "
                                                 f"'{model_type}' yet, so it needs an update first"))

    if safetensors:
        index = " + index" if "model.safetensors.index.json" in names else ""
        plural = "s" if len(safetensors) != 1 else ""
        items.append(Item("ok", "Weights", f"{len(safetensors)} safetensors file{plural}{index} · "
                                           f"{gb(sum(files[n] for n in safetensors))}"))
    elif bins:
        items.append(Item("warn", "Weights", f"old .bin format ({len(bins)} files): it loads, but it's slower and can "
                                             "run code, so safetensors is safer"))
    elif ggufs:
        items.append(Item("bad", "Weights", "only .gguf files, which can't be trained"))
    else:
        items.append(Item("bad", "Weights", "none found"))

    tokenizer = [n for n in TOKENIZER_FILES if n in names] or (["vocab.json", "merges.txt"]
                                                               if {"vocab.json", "merges.txt"} <= names else [])
    items.append(Item("ok" if tokenizer else "bad", "Tokenizer", ", ".join(tokenizer) or "no tokenizer files"))
    tokenizer_config = read_json("tokenizer_config.json") if "tokenizer_config.json" in names else None
    template = bool((tokenizer_config or {}).get("chat_template")) or bool(
        {"chat_template.jinja", "chat_template.json"} & names)
    items.append(Item("ok" if template else "info", "Chat template",
                      "found" if template else "none: fine for decision models and plain text, but chat training "
                                               "needs one"))

    decision = "joint_head.safetensors" in names
    if decision:
        items.append(Item("ok", "Decision head", "Clef joint schema head included"))

    verdict = _verdict(items)
    message = {"bad": "This model can't be trained as it is.",
               "warn": "This model can be trained, but read the notes first.",
               "ok": "Ready to train."}[verdict]
    return Report(verdict, message, "decision" if decision else "model", architecture or model_type, params, size,
                  items=items)


# ---------------------------------------------------------------- local folders

def safetensors_params(path: Path) -> int:
    """Number of values in a safetensors file, from its JSON header alone (the weights are never read)."""
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        if length > 100_000_000:
            raise ValueError(f"{path.name}: not a safetensors header")
        header = json.loads(f.read(length))
    total = 0
    for name, tensor in header.items():
        if name != "__metadata__":
            count = 1
            for dim in tensor["shape"]:
                count *= dim
            total += count
    return total


def check_folder(path: Path) -> Report:
    files = {p.name: p.stat().st_size for p in path.iterdir() if p.is_file()}

    def read_json(name: str) -> dict | None:
        try:
            return json.loads((path / name).read_text())
        except (OSError, ValueError):
            return None

    try:
        params = sum(safetensors_params(path / n) for n in files if n.endswith(".safetensors")) or None
    except (OSError, ValueError, KeyError):
        params = None
    return check(files, read_json, params)


# ---------------------------------------------------------------- the Hugging Face Hub

def _licence(card_data, tags: list[str]) -> str | None:
    licence = card_data.get("license") if card_data else None
    if isinstance(licence, list):
        licence = licence[0] if licence else None
    return licence or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)


def search(query: str, token: str | None, limit: int = 20) -> list[dict]:
    from huggingface_hub import HfApi

    models = HfApi().list_models(
        search=query or None, sort="downloads", limit=limit, token=token,
        expand=["downloads", "likes", "gated", "safetensors", "cardData", "tags", "pipeline_tag"])
    out = []
    for m in models:
        tags = list(m.tags or [])
        licence = _licence(m.card_data, tags)
        out.append({
            "id": m.id,
            "downloads": m.downloads or 0,
            "likes": m.likes or 0,
            "licence": licence,
            "permissive": licence in PERMISSIVE,
            "gated": bool(m.gated),
            "params": m.safetensors.total if m.safetensors else None,
            "pipeline": m.pipeline_tag,
            "custom_code": "custom_code" in tags,
            "gguf": "gguf" in tags,
        })
    return out


def check_repo(repo_id: str, token: str | None) -> Report:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        info = HfApi().model_info(repo_id, files_metadata=True, token=token)
    except GatedRepoError:
        info = None
    except RepositoryNotFoundError:
        return Report("bad", "There's no such model on the Hub, or it's private.", "unknown")
    if info is None:
        return Report("bad", "This model is gated.", "unknown", items=[Item("bad", "Access", "gated")])
    files = {s.rfilename: s.size or 0 for s in info.siblings or [] if "/" not in s.rfilename}

    def read_json(name: str) -> dict | None:
        try:
            return json.loads(Path(hf_hub_download(repo_id, name, token=token)).read_text())
        except GatedRepoError:
            raise
        except Exception:  # a broken or missing file reads as absent, like a local folder
            return None

    params = info.safetensors.total if getattr(info, "safetensors", None) else None
    try:
        report = check(files, read_json, params)
    except GatedRepoError:
        return Report("bad", "This model is gated. Accept its terms on its Hub page, then add your Hugging Face "
                             "token in Settings.", "unknown", params=params, size=sum(files.values()),
                      items=[Item("bad", "Access", "gated: your token doesn't have access yet")])
    if info.gated:
        report.items.insert(0, Item("info", "Access", "gated, and your token has access"))
    return report
