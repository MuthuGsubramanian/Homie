import pytest
from unittest.mock import MagicMock
from homie_core.model.engine import ModelEngine


def test_engine_not_loaded_raises():
    engine = ModelEngine()
    with pytest.raises(RuntimeError, match="No model loaded"):
        engine.generate("hello")


def test_engine_generate_with_mock_backend():
    engine = ModelEngine()
    mock_backend = MagicMock()
    mock_backend.generate.return_value = "Hello! How can I help?"
    engine._backend = mock_backend
    engine._loaded = True
    result = engine.generate("Say hello")
    assert result == "Hello! How can I help?"
    mock_backend.generate.assert_called_once()


def test_engine_stream_with_mock_backend():
    engine = ModelEngine()
    mock_backend = MagicMock()
    mock_backend.stream.return_value = iter(["Hello", " world"])
    engine._backend = mock_backend
    engine._loaded = True
    chunks = list(engine.stream("Say hello"))
    assert chunks == ["Hello", " world"]


def test_engine_unload():
    engine = ModelEngine()
    mock_backend = MagicMock()
    engine._backend = mock_backend
    engine._loaded = True
    engine.unload()
    assert not engine.is_loaded
    assert engine.current_model is None
    mock_backend.unload.assert_called_once()
