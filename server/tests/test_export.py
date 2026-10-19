"""Exporting a run as GGUF. The tiny test model's made-up tokenizer isn't one llama.cpp's converter knows, so stand-in
tools record what they were given; merging checkpoints, the files and the refusals are all real."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from anvil import config, runs
from test_training import (FORM, SERVER, decision_rows, finish, make_run, prepared, run_config, run_settings, start,
                           tiny_model)  # noqa: F401  (fixtures)

FAKE_CONVERTER = '''
import json, sys
from pathlib import Path
source = Path(sys.argv[1])
out = Path(sys.argv[sys.argv.index("--outfile") + 1])
config = json.loads((source / "config.json").read_text())
out.write_text(json.dumps({"files": sorted(p.name for p in source.iterdir()), "type": sys.argv[sys.argv.index("--outtype") + 1],
                           "architectures": config.get("architectures")}))
'''
FAKE_QUANTIZER = '''#!/bin/sh
python3 -c "import json, sys; d = json.load(open(sys.argv[1])); d['type'] = sys.argv[3]; json.dump(d, open(sys.argv[2], 'w'))" "$1" "$2" "$3"
'''


@pytest.fixture
def fake_llama(tmp_path) -> Path:
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text(FAKE_CONVERTER)
    quantize = root / "build" / "bin" / "llama-quantize"
    quantize.write_text(FAKE_QUANTIZER)
    quantize.chmod(0o755)
    return root


def run_export(run_dir: Path, version: str, quants: list[str], fake_llama: Path) -> dict:
    folder = run_dir / "exports" / version
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "request.json").write_text(json.dumps({"quants": quants}))
    env = {**os.environ, "LLAMA_CPP_DIR": str(fake_llama), "PYTHONPATH": str(SERVER), "HIP_VISIBLE_DEVICES": "",
           "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run([sys.executable, "-m", "anvil.export", str(run_dir), version], env=env, capture_output=True,
                          text=True, timeout=300)
    status = json.loads((folder / "status.json").read_text())
    assert done.returncode == 0, (status, done.stdout[-2000:], done.stderr[-2000:])
    return status


def gguf(folder: Path, name: str) -> dict:
    return json.loads((folder / name).read_text())


@pytest.fixture
def trained(tmp_path, tiny_model) -> Path:
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, maxSteps=4, every=2)
    cfg["checkpoints"].update(keepBest=5, patience=0)
    run_dir = make_run(tmp_path, cfg)
    finish(start(run_dir), run_dir)
    return run_dir


def test_the_finished_model_exports_in_the_types_asked_for(trained, fake_llama):
    status = run_export(trained, "final", ["Q8_0", "Q4_K_M"], fake_llama)
    folder = trained / "exports" / "final"
    assert status["state"] == "done" and [f["name"] for f in status["files"]] == ["test-final-Q8_0.gguf",
                                                                                 "test-final-Q4_K_M.gguf"]
    assert not (folder / "test-final-BF16.gguf").exists()  # the full-quality file was only a step on the way
    q4 = gguf(folder, "test-final-Q4_K_M.gguf")
    assert q4["type"] == "Q4_K_M" and "config.json" in q4["files"] and "tokenizer.json" in q4["files"]
    about = json.loads((folder / "anvil.json").read_text())
    picked = json.loads((trained / "model" / "merged" / "anvil.json").read_text())["picked_step"]
    assert about["version"] == "final" and about["step"] == picked and "auroc" in about["validation"]
    assert "Answer:" in about["use"]


def test_a_lora_checkpoint_is_merged_into_the_base_before_converting(trained, fake_llama, tiny_model):
    status = run_export(trained, "step-0000002", ["BF16"], fake_llama)
    folder = trained / "exports" / "step-0000002"
    full = gguf(folder, "test-step-0000002-BF16.gguf")
    assert status["state"] == "done" and full["type"] == "bf16"
    assert "model.safetensors" in full["files"] and "adapter_config.json" not in full["files"]  # merged, not an adapter
    assert not (folder / "hf").exists()  # the merged copy was temporary
    assert json.loads((folder / "anvil.json").read_text())["step"] == 2


def test_a_clef_checkpoint_exports_with_its_head(tmp_path, tiny_model, fake_llama):
    data = prepared(tmp_path, decision_rows(40), decision_rows(12, seed=1))
    cfg = run_config(tiny_model, data, maxSteps=4, every=2, maxLength=1024)
    cfg["method"] = "clef"
    cfg["checkpoints"].update(keepBest=5, patience=0)
    run_dir = make_run(tmp_path, cfg)
    finish(start(run_dir), run_dir)
    run_export(run_dir, "step-0000004", ["Q8_0"], fake_llama)
    q8 = gguf(run_dir / "exports" / "step-0000004", "test-step-0000004-Q8_0.gguf")
    assert {"joint_head_config.json", "joint_head.safetensors"} <= set(q8["files"])  # how llama.cpp knows it's Clef
    final = run_export(run_dir, "final", ["Q8_0"], fake_llama)
    assert {"joint_head_config.json", "joint_head.safetensors"} <= set(
        gguf(run_dir / "exports" / "final", final["files"][0]["name"])["files"])
    assert "systemone" in json.loads((run_dir / "exports" / "final" / "anvil.json").read_text())["use"]


def test_a_converter_failure_says_why(trained, tmp_path):
    broken = tmp_path / "broken-llama"
    (broken / "build" / "bin").mkdir(parents=True)
    (broken / "convert_hf_to_gguf.py").write_text("raise SystemExit('model architecture not supported: Tiny')")
    folder = trained / "exports" / "final"
    folder.mkdir(parents=True)
    (folder / "request.json").write_text(json.dumps({"quants": ["Q8_0"]}))
    env = {**os.environ, "LLAMA_CPP_DIR": str(broken), "PYTHONPATH": str(SERVER)}
    assert subprocess.run([sys.executable, "-m", "anvil.export", str(trained), "final"], env=env,
                          capture_output=True).returncode == 1
    status = json.loads((folder / "status.json").read_text())
    assert status["state"] == "failed" and "not supported: Tiny" in status["error"]


def test_starting_an_export_checks_what_it_can(run_settings, monkeypatch):
    run_id = runs.create(FORM, {"train": {}}, None)
    path = config.settings.data / "runs" / run_id
    (path / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid()}))
    with pytest.raises(ValueError, match="still going"):
        runs.start_export(run_id, "final", ["Q8_0"])
    (path / "status.json").write_text(json.dumps({"state": "done"}))
    with pytest.raises(ValueError, match="doesn't exist"):
        runs.start_export(run_id, "final", ["Q8_0"])  # no finished model in this made-up run
    (path / "checkpoints" / "step-0000010").mkdir(parents=True)
    (path / "checkpoints" / "step-0000010" / "state.json").write_text("{}")
    assert [v["version"] for v in runs.versions(run_id)] == ["step-0000010"]
    with pytest.raises(ValueError, match="at least one type"):
        runs.start_export(run_id, "step-0000010", ["Q3_FANCY"])
    started = []
    monkeypatch.setattr(runs.subprocess, "Popen", lambda *a, **k: started.append(a[0]) or type(
        "P", (), {"pid": os.getpid(), "poll": lambda self: None})())
    runs.start_export(run_id, "step-0000010", ["Q4_K_M", "Q8_0"])
    assert started[0][-2:] == [str(path), "step-0000010"]
    request = json.loads((path / "exports" / "step-0000010" / "request.json").read_text())
    assert request["quants"] == ["Q8_0", "Q4_K_M"]  # in the order of the list, whatever order they came in
    with pytest.raises(ValueError, match="still going"):
        runs.start_export(run_id, "step-0000010", ["Q8_0"])  # one export at a time
    with pytest.raises(ValueError, match="being exported"):
        runs.delete(run_id)


def test_only_an_exports_own_files_can_be_downloaded(run_settings):
    run_id = runs.create(FORM, {"train": {}}, None)
    folder = config.settings.data / "runs" / run_id / "exports" / "final"
    folder.mkdir(parents=True)
    (folder / "my-run-final-Q8_0.gguf").write_bytes(b"GGUF")
    (folder / "log.txt").write_text("secret-ish")
    assert runs.export_file(run_id, "final", "my-run-final-Q8_0.gguf").read_bytes() == b"GGUF"
    for version, name in (("final", "log.txt"), ("final", "../config.json"), ("..", "config.json"),
                          ("final", ".hidden.gguf"), ("final", "missing.gguf")):
        with pytest.raises(FileNotFoundError):
            runs.export_file(run_id, version, name)
