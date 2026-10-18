import json
import struct

import pytest

from anvil import config


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Test settings: the admin account (password "correct horse"), a data folder and one model root."""
    from anvil import accounts

    models = tmp_path / "models"
    models.mkdir()
    test_settings = config.Settings(password="correct horse", data=tmp_path / "data", model_roots=(models,),
                                    hf_token=None)
    monkeypatch.setattr(config, "settings", test_settings)
    accounts.ensure_default_user()
    return test_settings


def write_safetensors(path, shapes: dict[str, list[int]]):
    """A valid safetensors file with float32 zeros of the given shapes."""
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = 4
        for dim in shape:
            size *= dim
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))


def write_model(folder, model_type="llama", **extra):
    """A small but complete model folder: config, one safetensors file, a tokenizer and a chat template."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": model_type,
                                                    **extra}))
    write_safetensors(folder / "model.safetensors", {"embed": [10, 4], "head": [4, 10]})
    (folder / "tokenizer.json").write_text("{}")
    (folder / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    return folder
