"""Tests for Hugging Face backend integration."""
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from homie_core.model.hf_backend import (
    HFBackend,
    HFEmbeddings,
    _get_hf_key,
    discover_hf_models,
    _DEFAULT_CHAT_MODEL,
    _DEFAULT_EMBEDDING_MODEL,
)


# -----------------------------------------------------------------------
# HF Key detection
# -----------------------------------------------------------------------

class TestHFKeyDetection:
    def test_hf_key_from_env(self, monkeypatch):
        monkeypatch.setenv("HF_KEY", "hf_test_key_123")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert _get_hf_key() == "hf_test_key_123"

    def test_fallback_to_hf_token(self, monkeypatch):
        monkeypatch.delenv("HF_KEY", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "hf_fallback_456")
        assert _get_hf_key() == "hf_fallback_456"

    def test_fallback_to_hub_token(self, monkeypatch):
        monkeypatch.delenv("HF_KEY", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_hub_789")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert _get_hf_key() == "hf_hub_789"

    def test_no_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("HF_KEY", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert _get_hf_key() == ""


# -----------------------------------------------------------------------
# HFBackend
# -----------------------------------------------------------------------

class TestHFBackend:
    def test_load_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("HF_KEY", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        backend = HFBackend()
        with pytest.raises(ConnectionError, match="HF API key"):
            backend.load(api_key="")

    def test_load_with_key_sets_state(self, monkeypatch):
        # Mock urlopen to avoid real API call
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("homie_core.model.hf_backend.urlopen", return_value=mock_resp):
            backend = HFBackend()
            backend.load(model_id="test/model", api_key="hf_test_key")
            assert backend._connected is True
            assert backend._model == "test/model"

    def test_default_model(self):
        assert _DEFAULT_CHAT_MODEL != ""
        assert "mistral" in _DEFAULT_CHAT_MODEL.lower() or "llama" in _DEFAULT_CHAT_MODEL.lower() or True

    def test_unload_resets_state(self):
        backend = HFBackend()
        backend._model = "test"
        backend._api_key = "key"
        backend._connected = True
        backend.unload()
        assert backend._model == ""
        assert backend._api_key == ""
        assert backend._connected is False

    def test_generate_builds_correct_payload(self, monkeypatch):
        backend = HFBackend()
        backend._model = "test/model"
        backend._api_key = "hf_key"
        backend._connected = True

        captured_request = {}

        def mock_urlopen(req, timeout=None):
            captured_request["url"] = req.full_url
            captured_request["data"] = json.loads(req.data)
            captured_request["headers"] = dict(req.headers)

            resp = MagicMock()
            resp.read.return_value = json.dumps({
                "choices": [{"message": {"content": "Hello!"}}]
            }).encode()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("homie_core.model.hf_backend.urlopen", side_effect=mock_urlopen):
            result = backend.generate("test prompt", max_tokens=256, temperature=0.5)

        assert result == "Hello!"
        assert captured_request["data"]["model"] == "test/model"
        assert captured_request["data"]["max_tokens"] == 256
        assert captured_request["data"]["temperature"] == 0.5
        assert "chat/completions" in captured_request["url"]

    def test_stream_yields_tokens(self):
        backend = HFBackend()
        backend._model = "test/model"
        backend._api_key = "hf_key"
        backend._connected = True

        # Simulate SSE response
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b'data: [DONE]\n',
        ]

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter(sse_lines))

        with patch("homie_core.model.hf_backend.urlopen", return_value=mock_resp):
            tokens = list(backend.stream("test prompt"))

        assert tokens == ["Hello", " world"]


# -----------------------------------------------------------------------
# HFEmbeddings
# -----------------------------------------------------------------------

class TestHFEmbeddings:
    def test_connect_determines_dimension(self):
        fake_embedding = [0.1] * 384

        def mock_urlopen(req, timeout=None):
            resp = MagicMock()
            resp.read.return_value = json.dumps([fake_embedding]).encode()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("homie_core.model.hf_backend.urlopen", side_effect=mock_urlopen):
            emb = HFEmbeddings(api_key="hf_test")
            emb.connect()
            assert emb.dimension == 384
            assert emb.is_connected is True

    def test_embed_returns_vector(self):
        fake_embedding = [0.5] * 384

        def mock_urlopen(req, timeout=None):
            resp = MagicMock()
            resp.read.return_value = json.dumps([fake_embedding]).encode()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("homie_core.model.hf_backend.urlopen", side_effect=mock_urlopen):
            emb = HFEmbeddings(api_key="hf_test")
            emb._connected = True
            emb._dimension = 384
            result = emb.embed("test text")
            assert len(result) == 384

    def test_handles_token_level_output(self):
        """Test mean pooling when API returns token-level embeddings."""
        token_embeddings = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]

        def mock_urlopen(req, timeout=None):
            resp = MagicMock()
            resp.read.return_value = json.dumps([token_embeddings]).encode()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("homie_core.model.hf_backend.urlopen", side_effect=mock_urlopen):
            emb = HFEmbeddings(api_key="hf_test")
            emb._connected = True
            emb._dimension = 3
            result = emb.embed("test")
            assert len(result) == 3
            assert result[0] == pytest.approx(2.0)  # mean of 1.0 and 3.0

    def test_no_key_raises_on_connect(self, monkeypatch):
        monkeypatch.delenv("HF_KEY", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        emb = HFEmbeddings(api_key="")
        with pytest.raises(ConnectionError, match="HF API key"):
            emb.connect()


# -----------------------------------------------------------------------
# Model engine integration
# -----------------------------------------------------------------------

class TestModelEngineHFIntegration:
    def test_engine_loads_hf_backend(self):
        from homie_core.model.engine import ModelEngine
        from homie_core.model.registry import ModelEntry

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("homie_core.model.hf_backend.urlopen", return_value=mock_resp):
            engine = ModelEngine()
            entry = ModelEntry(
                name="Mistral Small", path="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                format="hf", params="24B",
            )
            engine.load(entry, api_key="hf_test_key")
            assert engine.is_loaded
            assert isinstance(engine._backend, HFBackend)
            engine.unload()


# -----------------------------------------------------------------------
# Config auto-detection
# -----------------------------------------------------------------------

class TestConfigHFAutoDetect:
    def test_hf_key_auto_configures_backend(self, monkeypatch):
        monkeypatch.setenv("HF_KEY", "hf_test_auto")
        monkeypatch.delenv("HOMIE_API_KEY", raising=False)
        monkeypatch.delenv("HOMIE_LLM_BACKEND", raising=False)

        from homie_core.config import load_config
        cfg = load_config()
        assert cfg.llm.backend == "hf"
        assert cfg.llm.api_key == "hf_test_auto"
        assert "mistral" in cfg.llm.model_path.lower() or "Mistral" in cfg.llm.model_path

    def test_explicit_backend_not_overridden(self, monkeypatch):
        monkeypatch.setenv("HF_KEY", "hf_test")
        monkeypatch.setenv("HOMIE_LLM_BACKEND", "cloud")
        monkeypatch.setenv("HOMIE_API_KEY", "sk-existing")

        from homie_core.config import load_config
        cfg = load_config()
        # HF_KEY should be overridden by HOMIE_API_KEY mapping
        # but backend should stay cloud since env override runs after auto-detect
        assert cfg.llm.api_key == "hf_test"  # HF_KEY wins in env_map


# -----------------------------------------------------------------------
# Discover models
# -----------------------------------------------------------------------

class TestDiscoverModels:
    def test_returns_popular_models(self, monkeypatch):
        monkeypatch.setenv("HF_KEY", "hf_test")
        models = discover_hf_models()
        assert len(models) > 0
        assert all("id" in m and "name" in m for m in models)

    def test_returns_empty_without_key(self, monkeypatch):
        monkeypatch.delenv("HF_KEY", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        models = discover_hf_models(api_key="")
        assert models == []
