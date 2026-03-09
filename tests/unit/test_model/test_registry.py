import pytest
from pathlib import Path
from homie_core.model.registry import ModelRegistry, ModelEntry


@pytest.fixture
def registry(tmp_path):
    r = ModelRegistry(tmp_path / "models")
    r.initialize()
    return r


def test_register_local_model(registry, tmp_path):
    model_file = tmp_path / "test.gguf"
    model_file.write_bytes(b"fake gguf data")
    registry.register("test-model", model_file, format="gguf", params="7B")
    entry = registry.get("test-model")
    assert entry is not None
    assert entry.name == "test-model"
    assert entry.format == "gguf"


def test_list_models(registry, tmp_path):
    f1 = tmp_path / "m1.gguf"
    f1.write_bytes(b"data")
    registry.register("model-a", f1, format="gguf", params="7B")
    registry.register("model-b", f1, format="gguf", params="13B")
    models = registry.list_models()
    assert len(models) == 2


def test_remove_model(registry, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"data")
    registry.register("to-delete", f, format="gguf", params="7B")
    registry.remove("to-delete")
    assert registry.get("to-delete") is None


def test_active_model(registry, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"data")
    registry.register("my-model", f, format="gguf", params="7B")
    registry.set_active("my-model")
    assert registry.get_active().name == "my-model"
