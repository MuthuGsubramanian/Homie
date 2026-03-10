"""Tests for ModelEngine cloud format dispatch."""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import ClassVar

import pytest

from homie_core.model.engine import ModelEngine
from homie_core.model.registry import ModelEntry


# ---------------------------------------------------------------------------
# Mock OpenAI-compatible HTTP server
# ---------------------------------------------------------------------------

class _MockHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible API handler for engine cloud tests."""

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = {"data": [{"id": "test-model"}]}
            self.wfile.write(json.dumps(body).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):  # noqa: N802
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)  # consume body
            response = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "cloud response"},
                        "finish_reason": "stop",
                    }
                ]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture(scope="module")
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_engine_loads_cloud_backend(mock_server):
    engine = ModelEngine()
    entry = ModelEntry(name="test-cloud", path="test-model", format="cloud", params="cloud")
    engine.load(entry, api_key="test-key", base_url=mock_server)
    assert engine.is_loaded
    assert engine.current_model.name == "test-cloud"
    engine.unload()


def test_engine_generate_cloud(mock_server):
    engine = ModelEngine()
    entry = ModelEntry(name="test-cloud", path="test-model", format="cloud", params="cloud")
    engine.load(entry, api_key="test-key", base_url=mock_server)
    result = engine.generate("Hello")
    assert result == "cloud response"
    engine.unload()


def test_engine_rejects_unknown_format():
    engine = ModelEngine()
    entry = ModelEntry(name="bad", path="bad-path", format="unknown", params="unknown")
    with pytest.raises(ValueError, match="Unsupported format: unknown"):
        engine.load(entry)
