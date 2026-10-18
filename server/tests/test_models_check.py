import json

from anvil.models_check import check_folder
from conftest import write_model, write_safetensors


def statuses(report):
    return {item.label: item.status for item in report.items}


def test_complete_folder_is_ready(tmp_path):
    report = check_folder(write_model(tmp_path / "m"))
    assert report.verdict == "ok"
    assert report.kind == "model"
    assert report.architecture == "LlamaForCausalLM"
    assert report.params == 80  # 10x4 + 4x10, read from the safetensors header alone


def test_gguf_only_folder_is_refused(tmp_path):
    folder = tmp_path / "gguf"
    folder.mkdir()
    (folder / "model.Q8_0.gguf").write_bytes(b"GGUF")
    report = check_folder(folder)
    assert report.verdict == "bad"
    assert "GGUF is for running" in report.message


def test_adapter_names_its_base_model(tmp_path):
    folder = tmp_path / "adapter"
    folder.mkdir()
    (folder / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 32, "base_model_name_or_path": "Qwen/Qwen3.5-2B-Base"}))
    write_safetensors(folder / "adapter_model.safetensors", {"lora_A": [32, 8]})
    report = check_folder(folder)
    assert report.verdict == "warn"
    assert report.kind == "adapter"
    assert report.base_model == "Qwen/Qwen3.5-2B-Base"


def test_unknown_architecture_needs_a_transformers_update(tmp_path):
    report = check_folder(write_model(tmp_path / "m", model_type="not_a_real_architecture"))
    assert report.verdict == "bad"
    assert statuses(report)["Architecture"] == "bad"


def test_custom_code_is_a_warning(tmp_path):
    report = check_folder(write_model(tmp_path / "m", model_type="custom",
                                      auto_map={"AutoModel": "modeling_custom.Model"}))
    assert report.verdict == "warn"
    assert statuses(report)["Custom code"] == "warn"


def test_missing_chat_template_is_only_a_note(tmp_path):
    folder = write_model(tmp_path / "m")
    (folder / "tokenizer_config.json").write_text("{}")
    report = check_folder(folder)
    assert report.verdict == "ok"
    assert statuses(report)["Chat template"] == "info"


def test_missing_tokenizer_is_refused(tmp_path):
    folder = write_model(tmp_path / "m")
    (folder / "tokenizer.json").unlink()
    assert check_folder(folder).verdict == "bad"


def test_clef_head_marks_a_decision_model(tmp_path):
    folder = write_model(tmp_path / "m")
    write_safetensors(folder / "joint_head.safetensors", {"w": [2, 2]})
    assert check_folder(folder).kind == "decision"
