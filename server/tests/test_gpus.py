from anvil import gpus


def fake_card(root, card, pci, total, used, connectors=()):
    """A sysfs card folder like /sys/class/drm/card1 -> .../0000:2d:00.0, with its connectors."""
    device = root / "devices" / pci
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "mem_info_vram_total").write_text(f"{total}\n")
    (device / "mem_info_vram_used").write_text(f"{used}\n")
    (root / "drm").mkdir(exist_ok=True)
    (root / "drm" / card).mkdir()
    (root / "drm" / card / "device").symlink_to(device)
    for name, status in connectors:
        (root / "drm" / f"{card}-{name}").mkdir()
        (root / "drm" / f"{card}-{name}" / "status").write_text(f"{status}\n")


def test_sysfs_reads_memory_and_screens(tmp_path):
    fake_card(tmp_path, "card0", "0000:30:00.0", 32 * gpus.GiB, 57 * 2**20, [("DP-5", "disconnected")])
    fake_card(tmp_path, "card1", "0000:2d:00.0", 32 * gpus.GiB, 2 * gpus.GiB, [("DP-1", "connected")])
    cards = gpus.read_sysfs(tmp_path / "drm")
    assert set(cards) == {"0000:30:00.0", "0000:2d:00.0"}
    assert cards["0000:2d:00.0"].display and not cards["0000:30:00.0"].display
    assert cards["0000:2d:00.0"].used == 2 * gpus.GiB


def test_runtime_label_names_rocm():
    assert gpus.runtime_label({"torch": "2.14.0+rocm10.1.0", "hip": "7.16"}) == ("rocm", "ROCm 10.1.0 · PyTorch 2.14.0")
    assert gpus.runtime_label({"torch": "2.14.0+cu130", "cuda": "13.0"}) == ("cuda", "CUDA 13.0 · PyTorch 2.14.0")


def test_busy_card_is_flagged(monkeypatch):
    monkeypatch.setattr(gpus, "probe", lambda: {"torch": "2.14.0+rocm10.1.0", "devices": [
        {"index": 0, "name": "AMD Radeon AI PRO R9700", "total": 32 * gpus.GiB, "pci": "0000:2d:00.0"}]})
    monkeypatch.setattr(gpus, "read_sysfs", lambda: {"0000:2d:00.0": gpus.Live(32 * gpus.GiB, 27 * gpus.GiB, False)})
    monkeypatch.setattr(gpus, "read_nvidia_smi", lambda: {})
    card = gpus.list_gpus()["gpus"][0]
    assert card["busy"] and card["used_gb"] == 27.0
