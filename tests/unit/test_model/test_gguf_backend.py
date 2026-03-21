"""Tests for GGUFBackend warm-up and load sequence."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from homie_core.model.gguf_backend import GGUFBackend


class TestWarmUp:
    def test_warm_up_calls_generate_with_minimal_args(self):
        backend = GGUFBackend()
        backend._base_url = "http://127.0.0.1:9999"
        with patch.object(backend, "generate", return_value="") as mock_gen:
            backend._warm_up()
        mock_gen.assert_called_once_with("Hi", max_tokens=1, temperature=0.0)

    def test_warm_up_swallows_exceptions(self):
        backend = GGUFBackend()
        backend._base_url = "http://127.0.0.1:9999"
        with patch.object(backend, "generate", side_effect=RuntimeError("server down")):
            # Should not raise
            backend._warm_up()

    def test_warm_up_swallows_connection_errors(self):
        backend = GGUFBackend()
        backend._base_url = "http://127.0.0.1:9999"
        with patch.object(backend, "generate", side_effect=OSError("connection refused")):
            backend._warm_up()


class TestLoadCallsWarmUp:
    def test_load_calls_warm_up_after_server_ready(self, tmp_path):
        """Verify load() invokes _warm_up() once after _wait_for_server() succeeds."""
        fake_exe = tmp_path / "llama-server.exe"
        fake_exe.write_text("fake")

        backend = GGUFBackend()
        fake_model = tmp_path / "model.gguf"
        fake_model.write_text("fake")

        with (
            patch("homie_core.model.gguf_backend._find_llama_server", return_value=str(fake_exe)),
            patch("homie_core.model.gguf_backend._find_free_port", return_value=19999),
            patch("homie_core.model.gguf_backend._find_cuda_libs", return_value=[]),
            patch("subprocess.Popen") as mock_popen,
            patch.object(backend, "_wait_for_server"),
            patch.object(backend, "_warm_up") as mock_warmup,
            patch("atexit.register"),
        ):
            mock_popen.return_value = MagicMock()
            backend.load(fake_model)

        mock_warmup.assert_called_once()

    def test_load_warm_up_called_after_wait(self, tmp_path):
        """Warm-up must happen AFTER _wait_for_server, not before."""
        fake_exe = tmp_path / "llama-server.exe"
        fake_exe.write_text("fake")

        backend = GGUFBackend()
        fake_model = tmp_path / "model.gguf"
        fake_model.write_text("fake")

        call_order = []

        def record_wait(*args, **kwargs):
            call_order.append("wait")

        def record_warmup(*args, **kwargs):
            call_order.append("warmup")

        with (
            patch("homie_core.model.gguf_backend._find_llama_server", return_value=str(fake_exe)),
            patch("homie_core.model.gguf_backend._find_free_port", return_value=19998),
            patch("homie_core.model.gguf_backend._find_cuda_libs", return_value=[]),
            patch("subprocess.Popen") as mock_popen,
            patch.object(backend, "_wait_for_server", side_effect=record_wait),
            patch.object(backend, "_warm_up", side_effect=record_warmup),
            patch("atexit.register"),
        ):
            mock_popen.return_value = MagicMock()
            backend.load(fake_model)

        assert call_order == ["wait", "warmup"]
