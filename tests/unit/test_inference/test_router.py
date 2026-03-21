"""Tests for the unified inference router."""
from unittest.mock import MagicMock, patch
import pytest

from homie_core.inference.router import InferenceRouter, _TIER_TIMEOUTS


def _make_config(priority=None):
    cfg = MagicMock()
    cfg.inference.priority = priority or ["local", "lan", "qubrid"]
    cfg.inference.qubrid.enabled = True
    cfg.inference.qubrid.model = "Qwen/Qwen3.5-Flash"
    cfg.inference.qubrid.base_url = "https://platform.qubrid.com/v1"
    cfg.inference.qubrid.timeout = 30
    cfg.inference.lan.prefer_desktop = True
    cfg.inference.lan.max_latency_ms = 500
    return cfg


def test_router_uses_local_first():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = "local response"
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    result = router.generate("hello")
    assert result == "local response"
    engine.generate.assert_called_once()


def test_router_falls_back_to_qubrid():
    engine = MagicMock()
    engine.is_loaded = False
    router = InferenceRouter(config=_make_config(), model_engine=engine, qubrid_api_key="test-key")
    with patch.object(router, "_qubrid") as mock_qubrid:
        mock_qubrid.is_available = True
        mock_qubrid.generate.return_value = "cloud response"
        result = router.generate("hello")
    assert result == "cloud response"


def test_router_all_sources_fail():
    engine = MagicMock()
    engine.is_loaded = False
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    with pytest.raises(RuntimeError, match="All inference sources unavailable"):
        router.generate("hello")


def test_router_active_source_local():
    engine = MagicMock()
    engine.is_loaded = True
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    assert router.active_source == "Local"


def test_router_active_source_qubrid():
    engine = MagicMock()
    engine.is_loaded = False
    router = InferenceRouter(config=_make_config(), model_engine=engine, qubrid_api_key="test-key")
    with patch.object(router, "_qubrid") as mock_qubrid:
        mock_qubrid.is_available = True
        assert router.active_source == "Homie Intelligence (Cloud)"


def test_router_fallback_banner_when_cloud():
    engine = MagicMock()
    engine.is_loaded = False
    router = InferenceRouter(config=_make_config(), model_engine=engine, qubrid_api_key="test-key")
    with patch.object(router, "_qubrid") as mock_qubrid:
        mock_qubrid.is_available = True
        assert router.fallback_banner == "No local model found! Using Homie's intelligence until local model is setup!"


def test_router_no_banner_when_local():
    engine = MagicMock()
    engine.is_loaded = True
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    assert router.fallback_banner is None


def test_router_stream_local():
    engine = MagicMock()
    engine.is_loaded = True
    engine.stream.return_value = iter(["chunk1", "chunk2"])
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    result = list(router.stream("hello"))
    assert result == ["chunk1", "chunk2"]


def test_router_stream_fallback_to_qubrid():
    engine = MagicMock()
    engine.is_loaded = False
    router = InferenceRouter(config=_make_config(), model_engine=engine, qubrid_api_key="test-key")
    with patch.object(router, "_qubrid") as mock_qubrid:
        mock_qubrid.is_available = True
        mock_qubrid.stream.return_value = iter(["cloud1", "cloud2"])
        result = list(router.stream("hello"))
    assert result == ["cloud1", "cloud2"]


def test_router_local_error_falls_through():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.side_effect = TimeoutError("Model timeout")
    router = InferenceRouter(config=_make_config(), model_engine=engine, qubrid_api_key="test-key")
    with patch.object(router, "_qubrid") as mock_qubrid:
        mock_qubrid.is_available = True
        mock_qubrid.generate.return_value = "cloud fallback"
        result = router.generate("hello")
    assert result == "cloud fallback"


# --- Tiered timeout tests ---

def test_tier_timeouts_dict_values():
    assert _TIER_TIMEOUTS["small"] == 8
    assert _TIER_TIMEOUTS["medium"] == 25
    assert _TIER_TIMEOUTS["large"] == 90


def test_generate_small_tier_uses_8s_timeout():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = "ok"
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    router.generate("hi", tier="small")
    _, kwargs = engine.generate.call_args
    assert kwargs["timeout"] == 8


def test_generate_medium_tier_uses_25s_timeout():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = "ok"
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    router.generate("hi", tier="medium")
    _, kwargs = engine.generate.call_args
    assert kwargs["timeout"] == 25


def test_generate_large_tier_uses_90s_timeout():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = "ok"
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    router.generate("hi", tier="large")
    _, kwargs = engine.generate.call_args
    assert kwargs["timeout"] == 90


def test_generate_no_tier_uses_default_timeout():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = "ok"
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    router.generate("hi", timeout=55)
    _, kwargs = engine.generate.call_args
    assert kwargs["timeout"] == 55


def test_generate_unknown_tier_falls_back_to_default():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = "ok"
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    router.generate("hi", timeout=77, tier="xlarge")
    _, kwargs = engine.generate.call_args
    assert kwargs["timeout"] == 77


def test_generate_tier_none_does_not_override_timeout():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = "ok"
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    router.generate("hi", timeout=42, tier=None)
    _, kwargs = engine.generate.call_args
    assert kwargs["timeout"] == 42
