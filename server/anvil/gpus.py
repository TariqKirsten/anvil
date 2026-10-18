"""The GPUs on this machine: their names, how much memory each one uses right now, and which one drives a screen.

Two sources, because neither knows everything:
- A short PyTorch probe in a child process gives the names and the numbering that training will use (the index PyTorch
  gives each card). It runs once, so the server itself never holds a GPU.
- sysfs (AMD) or nvidia-smi (NVIDIA) gives the live memory numbers and the screen connections. Reading them never
  touches the GPU, so it's cheap enough to refresh every few seconds.
The two are matched by PCI address.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SYSFS = Path("/sys/class/drm")
GiB = 1024**3
BUSY_SHARE = 0.25  # another program holding at least this share of a card's memory makes it "busy"

PROBE = r"""
import json, torch
devices = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    bus = getattr(p, "pci_bus_id", None)
    pci = None if bus is None else "%04x:%02x:%02x.0" % (getattr(p, "pci_domain_id", 0), bus, getattr(p, "pci_device_id", 0))
    devices.append({"index": i, "name": p.name, "total": p.total_memory, "pci": pci, "arch": getattr(p, "gcnArchName", None)})
print(json.dumps({"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda, "devices": devices}))
"""


@dataclass
class Live:
    total: int
    used: int
    display: bool


def read_sysfs(root: Path = SYSFS) -> dict[str, Live]:
    """Live memory and screen connections of every AMD card, by PCI address (e.g. '0000:2d:00.0')."""
    cards = {}
    for card in sorted(root.glob("card[0-9]*")):
        if "-" in card.name:  # a connector such as card1-DP-1, not a card
            continue
        device = card / "device"
        try:
            if (device / "vendor").read_text().strip() != "0x1002":
                continue
            total = int((device / "mem_info_vram_total").read_text())
            used = int((device / "mem_info_vram_used").read_text())
        except (OSError, ValueError):
            continue
        display = any(c.joinpath("status").read_text().strip() == "connected"
                      for c in root.glob(f"{card.name}-*") if c.joinpath("status").is_file())
        cards[device.resolve().name.lower()] = Live(total, used, display)
    return cards


def read_nvidia_smi() -> dict[str, Live]:
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,memory.total,memory.used,display_active",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    cards = {}
    for line in out.strip().splitlines():
        bus, total, used, display = (x.strip() for x in line.split(","))
        cards[bus.lower()[-12:]] = Live(int(total) * 2**20, int(used) * 2**20, display.lower() == "enabled")
    return cards


def read_rocm_smi_names() -> dict[str, str]:
    """Marketing names by PCI address, for when PyTorch only knows a generic name."""
    if not shutil.which("rocm-smi"):
        return {}
    try:
        out = subprocess.run(["rocm-smi", "--showbus", "--showproductname", "--json"],
                             capture_output=True, text=True, timeout=20).stdout
        data = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    return {v["PCI Bus"].lower(): v["Card Series"] for v in data.values()
            if isinstance(v, dict) and v.get("PCI Bus") and v.get("Card Series")}


_lock = threading.Lock()
_probe: dict | None = None
_probe_failed_at = 0.0


def probe() -> dict:
    """What PyTorch sees: its version, the ROCm/CUDA build, and each card. Cached after the first success."""
    global _probe, _probe_failed_at
    with _lock:
        if _probe is not None:
            return _probe
        if time.time() - _probe_failed_at < 30:
            return {"error": "PyTorch couldn't see the GPUs.", "devices": []}
        env = {k: v for k, v in os.environ.items()
               if k not in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
        try:
            result = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                                    timeout=120, env=env)
            found = json.loads(result.stdout.strip().splitlines()[-1])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            _probe_failed_at = time.time()
            return {"error": "PyTorch couldn't see the GPUs.", "devices": []}
        _probe = found
        return found


def runtime_label(found: dict) -> tuple[str, str]:
    torch_version = found.get("torch") or ""
    match = re.search(r"\+rocm(\d+(?:\.\d+)*)", torch_version)
    base = torch_version.split("+")[0]
    if match or found.get("hip"):
        return "rocm", f"ROCm {match.group(1) if match else found['hip']} · PyTorch {base}"
    if found.get("cuda"):
        return "cuda", f"CUDA {found['cuda']} · PyTorch {base}"
    return "none", f"PyTorch {base}" if base else "No GPU runtime"


def list_gpus() -> dict:
    found = probe()
    live = read_sysfs() | read_nvidia_smi()
    names = None
    gpus = []
    for d in found.get("devices", []):
        card = live.get(d["pci"]) if d.get("pci") else None
        name = d["name"]
        if name in ("", "AMD Radeon Graphics"):
            names = read_rocm_smi_names() if names is None else names
            name = names.get(d.get("pci") or "", name)
        total = card.total if card else d["total"]
        used = card.used if card else None
        gpus.append({
            "index": d["index"],
            "name": name,
            "arch": d.get("arch"),
            "pci": d.get("pci"),
            "total_gb": round(total / GiB, 1),
            "used_gb": None if used is None else round(used / GiB, 1),
            "display": bool(card and card.display),
            "busy": bool(used is not None and used >= BUSY_SHARE * total),
        })
    kind, label = runtime_label(found)
    return {"runtime": {"kind": kind, "label": label, "error": found.get("error")}, "gpus": gpus}
