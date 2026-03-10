# Cloud API Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an OpenAI-compatible cloud API backend so Homie can use any cloud provider (OpenAI, Groq, Together, DeepSeek, OpenRouter, etc.) as an alternative to local GGUF inference.

**Architecture:** A new `CloudBackend` class implements the same `load/generate/stream/unload` interface as `GGUFBackend`. The `ModelEngine` dispatches to it when `format="cloud"`. The setup wizard branches after hardware detection to ask local vs cloud. API keys are stored in `homie.config.yaml`. No new dependencies — uses stdlib `urllib`.

**Tech Stack:** Python stdlib (`urllib.request`, `json`), Pydantic config, pytest with mock HTTP server

---

### Task 1: CloudBackend — Core Implementation

**Files:**
- Create: `src/homie_core/model/cloud_backend.py`
- Test: `tests/unit/test_model/test_cloud_backend.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_model/test_cloud_backend.py`:

```python
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import pytest

from homie_core.model.cloud_backend import CloudBackend


class MockCloudHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            resp = {"data": [
                {"id": "gpt-4o"},
                {"id": "gpt-4o-mini"},
                {"id": "text-embedding-3-small"},
            ]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            # Check auth header
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": "Invalid API key"}}).encode())
                return

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for word in ["Hello", " from", " cloud", "!"]:
                    chunk = {
                        "choices": [{"delta": {"content": word}, "index": 0}]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                resp = {
                    "choices": [{
                        "message": {"role": "assistant", "content": "Hello from cloud!"},
                        "finish_reason": "stop",
                    }]
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


@pytest.fixture
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockCloudHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


def test_load_connects_and_discovers_models(mock_server):
    backend = CloudBackend()
    backend.load(model_id="gpt-4o", api_key="test-key", base_url=mock_server)
    assert backend._connected
    assert backend._model == "gpt-4o"


def test_load_connection_failure():
    backend = CloudBackend()
    with pytest.raises(ConnectionError, match="Cannot connect"):
        backend.load(model_id="gpt-4o", api_key="test-key", base_url="http://127.0.0.1:59999/v1")


def test_generate(mock_server):
    backend = CloudBackend()
    backend.load(model_id="gpt-4o", api_key="test-key", base_url=mock_server)
    result = backend.generate("Hello")
    assert result == "Hello from cloud!"


def test_generate_auth_failure(mock_server):
    backend = CloudBackend()
    backend.load(model_id="gpt-4o", api_key="test-key", base_url=mock_server)
    backend._api_key = ""  # break the key
    with pytest.raises(RuntimeError, match="API error"):
        backend.generate("Hello")


def test_stream(mock_server):
    backend = CloudBackend()
    backend.load(model_id="gpt-4o", api_key="test-key", base_url=mock_server)
    chunks = list(backend.stream("Hello"))
    assert "".join(chunks) == "Hello from cloud!"


def test_unload(mock_server):
    backend = CloudBackend()
    backend.load(model_id="gpt-4o", api_key="test-key", base_url=mock_server)
    assert backend._connected
    backend.unload()
    assert not backend._connected


def test_discover_models(mock_server):
    backend = CloudBackend()
    models = backend.discover_models(api_key="test-key", base_url=mock_server)
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models


def test_discover_models_failure():
    backend = CloudBackend()
    models = backend.discover_models(api_key="test-key", base_url="http://127.0.0.1:59999/v1")
    assert models == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_model/test_cloud_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'homie_core.model.cloud_backend'`

**Step 3: Write the implementation**

Create `src/homie_core/model/cloud_backend.py`:

```python
from __future__ import annotations

import json
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class CloudBackend:
    """Backend that connects to any OpenAI-compatible cloud API."""

    def __init__(self):
        self._base_url: str = ""
        self._model: str = ""
        self._api_key: str = ""
        self._connected = False

    def load(self, model_id: str, api_key: str, base_url: str = "https://api.openai.com/v1", **kwargs) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model_id
        self._api_key = api_key

        # Verify connection
        try:
            req = Request(f"{self._base_url}/models", method="GET")
            req.add_header("Authorization", f"Bearer {self._api_key}")
            with urlopen(req, timeout=10) as resp:
                json.loads(resp.read())
            self._connected = True
        except (URLError, TimeoutError, OSError) as e:
            raise ConnectionError(
                f"Cannot connect to cloud API at {self._base_url}. "
                f"Check your endpoint URL and API key. Error: {e}"
            )

    def discover_models(self, api_key: str, base_url: str = "https://api.openai.com/v1") -> list[str]:
        """Query the /v1/models endpoint and return available model IDs."""
        try:
            req = Request(f"{base_url.rstrip('/')}/models", method="GET")
            req.add_header("Authorization", f"Bearer {api_key}")
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                models = data.get("data", [])
                return [m.get("id", "") for m in models if m.get("id")]
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            return []

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        req = Request(f"{self._base_url}/chat/completions", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            with urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"]
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(body)
                msg = err.get("error", {}).get("message", body)
            except json.JSONDecodeError:
                msg = body
            raise RuntimeError(f"API error ({e.code}): {msg}")
        except URLError as e:
            raise RuntimeError(f"Connection error: {e}")

    def stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> Iterator[str]:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        req = Request(f"{self._base_url}/chat/completions", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")

        with urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def unload(self) -> None:
        self._connected = False
        self._base_url = ""
        self._model = ""
        self._api_key = ""
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_model/test_cloud_backend.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/homie_core/model/cloud_backend.py tests/unit/test_model/test_cloud_backend.py
git commit -m "feat: add CloudBackend for OpenAI-compatible cloud APIs"
```

---

### Task 2: Config — Add Cloud Fields

**Files:**
- Modify: `src/homie_core/config.py:11-18` (LLMConfig) and `src/homie_core/config.py:67-74` (env_map)
- Test: `tests/unit/test_config.py` (check if exists, else `tests/unit/test_config_cloud.py`)

**Step 1: Write the failing tests**

Create `tests/unit/test_config_cloud.py`:

```python
import os
import tempfile
from pathlib import Path

import yaml

from homie_core.config import HomieConfig, LLMConfig, load_config


def test_llm_config_has_cloud_fields():
    cfg = LLMConfig()
    assert hasattr(cfg, "api_key")
    assert hasattr(cfg, "api_base_url")
    assert cfg.api_key == ""
    assert cfg.api_base_url == ""


def test_cloud_config_roundtrip():
    cfg = HomieConfig()
    cfg.llm.backend = "cloud"
    cfg.llm.model_path = "gpt-4o"
    cfg.llm.api_key = "sk-test-123"
    cfg.llm.api_base_url = "https://api.openai.com/v1"

    data = cfg.model_dump()
    restored = HomieConfig(**data)
    assert restored.llm.backend == "cloud"
    assert restored.llm.model_path == "gpt-4o"
    assert restored.llm.api_key == "sk-test-123"
    assert restored.llm.api_base_url == "https://api.openai.com/v1"


def test_cloud_config_yaml_roundtrip(tmp_path):
    cfg = HomieConfig()
    cfg.llm.backend = "cloud"
    cfg.llm.model_path = "gpt-4o"
    cfg.llm.api_key = "sk-test-456"
    cfg.llm.api_base_url = "https://api.groq.com/openai/v1"

    config_file = tmp_path / "homie.config.yaml"
    config_file.write_text(yaml.dump(cfg.model_dump(), default_flow_style=False))

    loaded = load_config(config_file)
    assert loaded.llm.backend == "cloud"
    assert loaded.llm.api_key == "sk-test-456"
    assert loaded.llm.api_base_url == "https://api.groq.com/openai/v1"


def test_api_key_env_override(tmp_path, monkeypatch):
    cfg = HomieConfig()
    cfg.llm.backend = "cloud"
    cfg.llm.api_key = "from-config"

    config_file = tmp_path / "homie.config.yaml"
    config_file.write_text(yaml.dump(cfg.model_dump(), default_flow_style=False))

    monkeypatch.setenv("HOMIE_API_KEY", "from-env")
    loaded = load_config(config_file)
    assert loaded.llm.api_key == "from-env"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_cloud.py -v`
Expected: FAIL — `api_key` and `api_base_url` fields don't exist yet

**Step 3: Update LLMConfig and env overrides**

In `src/homie_core/config.py`, modify `LLMConfig` (line 11-18) — add two fields:

```python
class LLMConfig(BaseModel):
    model_path: str = str(Path.home() / ".lmstudio" / "models" / "lmstudio-community" / "Qwen3.5-35B-A3B-GGUF" / "Qwen3.5-35B-A3B-Q4_K_M.gguf")
    backend: str = "gguf"
    context_length: int = 65536
    temperature: float = 0.7
    max_tokens: int = 2048
    gpu_layers: int = -1
    repo_id: str = "lmstudio-community/Qwen3.5-35B-A3B-GGUF"
    api_key: str = ""
    api_base_url: str = ""
```

In `_apply_env_overrides` (line 67-74), add to `env_map`:

```python
    env_map = {
        "HOMIE_LLM_BACKEND": ("llm", "backend"),
        "HOMIE_LLM_MODEL_PATH": ("llm", "model_path"),
        "HOMIE_LLM_GPU_LAYERS": ("llm", "gpu_layers"),
        "HOMIE_API_KEY": ("llm", "api_key"),
        "HOMIE_API_BASE_URL": ("llm", "api_base_url"),
        "HOMIE_VOICE_ENABLED": ("voice", "enabled"),
        "HOMIE_STORAGE_PATH": ("storage", "path"),
        "HOMIE_USER_NAME": ("user_name",),
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_cloud.py -v`
Expected: All 4 tests PASS

**Step 5: Run all existing tests to check for regressions**

Run: `pytest tests/ -q`
Expected: All 200+ tests PASS

**Step 6: Commit**

```bash
git add src/homie_core/config.py tests/unit/test_config_cloud.py
git commit -m "feat: add api_key and api_base_url fields to LLMConfig"
```

---

### Task 3: ModelEngine — Dispatch to CloudBackend

**Files:**
- Modify: `src/homie_core/model/engine.py:23-35` (load method)
- Test: `tests/unit/test_model/test_engine_cloud.py`

**Step 1: Write the failing test**

Create `tests/unit/test_model/test_engine_cloud.py`:

```python
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import pytest

from homie_core.model.engine import ModelEngine
from homie_core.model.registry import ModelEntry


class MockCloudAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "test-model"}]}).encode())

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            resp = {"choices": [{"message": {"role": "assistant", "content": "cloud response"}, "finish_reason": "stop"}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())

    def log_message(self, format, *args):
        pass


@pytest.fixture
def mock_cloud():
    server = HTTPServer(("127.0.0.1", 0), MockCloudAPIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


def test_engine_loads_cloud_backend(mock_cloud):
    engine = ModelEngine()
    entry = ModelEntry(name="test-cloud", path="test-model", format="cloud", params="cloud")
    engine.load(entry, api_key="test-key", base_url=mock_cloud)
    assert engine.is_loaded
    assert engine.current_model.name == "test-cloud"


def test_engine_generate_cloud(mock_cloud):
    engine = ModelEngine()
    entry = ModelEntry(name="test-cloud", path="test-model", format="cloud", params="cloud")
    engine.load(entry, api_key="test-key", base_url=mock_cloud)
    result = engine.generate("Hello")
    assert result == "cloud response"
    engine.unload()


def test_engine_rejects_unknown_format():
    engine = ModelEngine()
    entry = ModelEntry(name="bad", path="bad", format="unknown", params="?")
    with pytest.raises(ValueError, match="Unsupported format"):
        engine.load(entry)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_model/test_engine_cloud.py -v`
Expected: FAIL — `format="cloud"` raises `ValueError: Unsupported format: cloud`

**Step 3: Add cloud dispatch to ModelEngine.load()**

In `src/homie_core/model/engine.py`, modify the `load` method (line 23-35):

```python
    def load(self, entry: ModelEntry, n_ctx: int = 4096, n_gpu_layers: int = -1, **kwargs) -> None:
        if entry.format == "gguf":
            from homie_core.model.gguf_backend import GGUFBackend
            backend = GGUFBackend()
            backend.load(entry.path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
            self._backend = backend
        elif entry.format == "safetensors":
            from homie_core.model.transformers_backend import TransformersBackend
            backend = TransformersBackend()
            backend.load(entry.path)
            self._backend = backend
        elif entry.format == "cloud":
            from homie_core.model.cloud_backend import CloudBackend
            backend = CloudBackend()
            backend.load(
                model_id=entry.path,
                api_key=kwargs.get("api_key", ""),
                base_url=kwargs.get("base_url", "https://api.openai.com/v1"),
            )
            self._backend = backend
        else:
            raise ValueError(f"Unsupported format: {entry.format}")
        self._loaded = True
        self._current_model = entry
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_model/test_engine_cloud.py -v`
Expected: All 3 tests PASS

**Step 5: Run all tests**

Run: `pytest tests/ -q`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/homie_core/model/engine.py tests/unit/test_model/test_engine_cloud.py
git commit -m "feat: add cloud format dispatch to ModelEngine"
```

---

### Task 4: CLI — Wire Cloud Config Into Model Loading

**Files:**
- Modify: `src/homie_app/cli.py:61-90` (_load_model_engine function)
- Modify: `src/homie_app/cli.py:32` (add "cloud" to format choices)

**Step 1: Update `_load_model_engine` to pass cloud kwargs**

In `src/homie_app/cli.py`, replace `_load_model_engine` (line 61-90):

```python
def _load_model_engine(cfg):
    """Load the model engine from local file or cloud API."""
    from homie_core.model.engine import ModelEngine
    from homie_core.model.registry import ModelRegistry, ModelEntry

    engine = ModelEngine()

    # Check registry for an active model first
    registry = ModelRegistry(Path(cfg.storage.path) / cfg.storage.models_dir)
    registry.initialize()

    entry = registry.get_active()

    if not entry and cfg.llm.model_path:
        entry = ModelEntry(
            name=cfg.llm.model_path if cfg.llm.backend == "cloud" else "Qwen3.5-35B-A3B",
            path=cfg.llm.model_path,
            format=cfg.llm.backend,
            params="cloud" if cfg.llm.backend == "cloud" else "35B-A3B",
        )

    if not entry:
        return None, None

    if entry.format == "cloud":
        print(f"  Connecting to cloud API: {entry.path}")
        print(f"  Endpoint: {cfg.llm.api_base_url or 'https://api.openai.com/v1'}")
        engine.load(entry, api_key=cfg.llm.api_key, base_url=cfg.llm.api_base_url or "https://api.openai.com/v1")
        print(f"  Connected!")
    else:
        print(f"  Loading model: {entry.name} ({entry.format})")
        print(f"  Path: {entry.path}")
        print(f"  Context: {cfg.llm.context_length:,} tokens")
        engine.load(entry, n_ctx=cfg.llm.context_length, n_gpu_layers=cfg.llm.gpu_layers)

    print(f"  Model loaded successfully!")
    return engine, entry
```

**Step 2: Add "cloud" to format choices in argparse**

In `src/homie_app/cli.py`, line 32, change:

```python
    add.add_argument("--format", type=str, default="gguf", choices=["gguf", "safetensors", "cloud"])
```

**Step 3: Run all tests**

Run: `pytest tests/ -q`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/homie_app/cli.py
git commit -m "feat: wire cloud config into CLI model loading"
```

---

### Task 5: Setup Wizard — Cloud Branch

**Files:**
- Modify: `src/homie_app/init.py` (full rewrite of run_init)
- Test: `tests/unit/test_init_cloud.py`

**Step 1: Write the failing test**

Create `tests/unit/test_init_cloud.py`:

```python
import yaml
from pathlib import Path
from unittest.mock import patch

from homie_core.config import HomieConfig


def test_cloud_setup_saves_config(tmp_path, monkeypatch):
    """Simulate user choosing cloud setup."""
    monkeypatch.chdir(tmp_path)

    # Simulate user inputs: "2" (cloud), "1" (OpenAI), "sk-test", "1" (first model), "TestUser"
    inputs = iter(["2", "1", "sk-test-key", "1", "TestUser"])

    from homie_app.init import run_init

    with patch("builtins.input", lambda prompt="": next(inputs)):
        with patch("homie_app.init.detect_hardware") as mock_hw:
            mock_hw.return_value.os_name = "Windows"
            mock_hw.return_value.os_version = "11"
            mock_hw.return_value.cpu_cores = 8
            mock_hw.return_value.ram_gb = 32
            mock_hw.return_value.gpus = []
            mock_hw.return_value.has_microphone = False
            mock_hw.return_value.best_gpu_vram_gb = 0

            with patch("homie_app.init._discover_cloud_models", return_value=["gpt-4o", "gpt-4o-mini"]):
                cfg = run_init(auto=False)

    assert cfg.llm.backend == "cloud"
    assert cfg.llm.api_key == "sk-test-key"
    assert cfg.llm.model_path == "gpt-4o"
    assert cfg.user_name == "TestUser"

    # Verify YAML was written
    config_file = tmp_path / "homie.config.yaml"
    assert config_file.exists()
    saved = yaml.safe_load(config_file.read_text())
    assert saved["llm"]["backend"] == "cloud"


def test_local_setup_unchanged(tmp_path, monkeypatch):
    """Simulate user choosing local setup — existing flow should still work."""
    monkeypatch.chdir(tmp_path)

    inputs = iter(["1", "TestUser"])

    from homie_app.init import run_init

    with patch("builtins.input", lambda prompt="": next(inputs)):
        with patch("homie_app.init.detect_hardware") as mock_hw:
            mock_hw.return_value.os_name = "Windows"
            mock_hw.return_value.os_version = "11"
            mock_hw.return_value.cpu_cores = 8
            mock_hw.return_value.ram_gb = 32
            mock_hw.return_value.gpus = []
            mock_hw.return_value.has_microphone = False
            mock_hw.return_value.best_gpu_vram_gb = 0

            with patch("homie_app.init.discover_local_model", return_value=None):
                cfg = run_init(auto=False)

    assert cfg.llm.backend == "gguf"
    assert cfg.llm.api_key == ""
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_init_cloud.py -v`
Expected: FAIL — `_discover_cloud_models` doesn't exist, setup doesn't ask local/cloud

**Step 3: Rewrite `init.py` with cloud branch**

Replace `src/homie_app/init.py` entirely:

```python
from __future__ import annotations

from pathlib import Path

from homie_core.config import HomieConfig, load_config
from homie_core.hardware.detector import detect_hardware
from homie_core.hardware.profiles import recommend_model, discover_local_model
from homie_core.model.registry import ModelRegistry

import yaml


# Known cloud providers with their API endpoints
CLOUD_PROVIDERS = [
    {"name": "OpenAI", "base_url": "https://api.openai.com/v1"},
    {"name": "Groq", "base_url": "https://api.groq.com/openai/v1"},
    {"name": "Together", "base_url": "https://api.together.xyz/v1"},
    {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1"},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1"},
]

# Fallback model presets when /v1/models discovery fails
PRESET_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "claude-sonnet-4",
    "deepseek-chat",
    "llama-3.1-70b",
    "mixtral-8x7b",
]


def _discover_cloud_models(api_key: str, base_url: str) -> list[str]:
    """Try to discover available models from the cloud API."""
    from homie_core.model.cloud_backend import CloudBackend
    backend = CloudBackend()
    return backend.discover_models(api_key=api_key, base_url=base_url)


def _ask_choice(prompt: str, options: list[str]) -> int:
    """Ask the user to pick from a numbered list. Returns 0-based index."""
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            raw = input("  > ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def _setup_cloud(cfg: HomieConfig) -> None:
    """Interactive cloud API configuration."""
    # Pick provider
    provider_names = [p["name"] for p in CLOUD_PROVIDERS] + ["Custom endpoint"]
    idx = _ask_choice("\n  Select cloud provider:", provider_names)

    if idx < len(CLOUD_PROVIDERS):
        base_url = CLOUD_PROVIDERS[idx]["base_url"]
        print(f"  Endpoint: {base_url}")
    else:
        base_url = input("  Enter API endpoint URL (e.g. https://api.example.com/v1): ").strip()

    cfg.llm.api_base_url = base_url

    # Get API key
    api_key = input("  Enter your API key: ").strip()
    cfg.llm.api_key = api_key

    # Discover or pick model
    print("  Discovering available models...")
    models = _discover_cloud_models(api_key, base_url)

    if models:
        # Filter out embedding models
        chat_models = [m for m in models if "embed" not in m.lower()]
        if not chat_models:
            chat_models = models
        # Show up to 15 models
        display_models = chat_models[:15]
        if len(chat_models) > 15:
            display_models.append(f"... and {len(chat_models) - 15} more")
        display_models.append("Enter model name manually")
        idx = _ask_choice("  Select a model:", display_models)

        if idx < len(chat_models) and idx < 15:
            cfg.llm.model_path = chat_models[idx]
        else:
            cfg.llm.model_path = input("  Enter model name: ").strip()
    else:
        print("  Could not auto-discover models. Showing presets.")
        preset_options = PRESET_MODELS + ["Enter model name manually"]
        idx = _ask_choice("  Select a model:", preset_options)

        if idx < len(PRESET_MODELS):
            cfg.llm.model_path = PRESET_MODELS[idx]
        else:
            cfg.llm.model_path = input("  Enter model name: ").strip()

    cfg.llm.backend = "cloud"
    print(f"  Selected: {cfg.llm.model_path}")


def run_init(auto: bool = False, config_path: str | None = None) -> HomieConfig:
    print("=" * 50)
    print("  Welcome to Homie AI Setup")
    print("=" * 50)

    # Step 1: Detect hardware
    print("\n[1/7] Detecting hardware...")
    hw = detect_hardware()
    print(f"  OS: {hw.os_name} {hw.os_version}")
    print(f"  CPU: {hw.cpu_cores} cores")
    print(f"  RAM: {hw.ram_gb} GB")
    if hw.gpus:
        for gpu in hw.gpus:
            print(f"  GPU: {gpu.name} ({gpu.vram_mb} MB VRAM)")
    else:
        print("  GPU: None detected")
    print(f"  Microphone: {'Yes' if hw.has_microphone else 'No'}")

    cfg = HomieConfig()

    # Step 2: Local or Cloud?
    if not auto:
        idx = _ask_choice(
            "\n[2/7] How would you like to run your AI model?",
            ["Local model (GGUF on this machine)", "Cloud API (OpenAI, Groq, Together, etc.)"],
        )
        use_cloud = idx == 1
    else:
        use_cloud = False

    if use_cloud:
        # Step 3: Cloud setup
        print("\n[3/7] Cloud API configuration...")
        _setup_cloud(cfg)

        # Step 4: Skip local model discovery
        print("\n[4/7] Skipping local model search (using cloud).")
        model_path = None
        rec = None
    else:
        # Step 3: Recommend model
        print("\n[3/7] Selecting optimal model...")
        rec = recommend_model(hw.best_gpu_vram_gb)
        print(f"  Recommended: {rec['model']} ({rec['quant']}, {rec['format']})")
        print(f"  Backend: {rec['backend']}")
        print(f"  Context length: {rec['context_length']:,} tokens")

        # Step 4: Discover existing local model files
        print("\n[4/7] Searching for existing model files...")
        model_path = discover_local_model(rec["model"])
        if model_path:
            print(f"  Found: {model_path}")
        else:
            print(f"  No local copy of {rec['model']} found.")
            print(f"  You can download it later with: homie model download {rec.get('repo_id', '')}")

        cfg.llm.backend = rec["format"]
        cfg.llm.context_length = rec["context_length"]
        cfg.llm.repo_id = rec.get("repo_id", "")
        if model_path:
            cfg.llm.model_path = model_path
        print(f"  Backend: {rec['format']} (direct loading via llama-server)")

    # Step 5: Voice setup
    cfg.voice.enabled = hw.has_microphone
    if hw.has_microphone:
        cfg.voice.mode = "push_to_talk"
    print(f"\n[5/7] Voice: {'push-to-talk' if hw.has_microphone else 'disabled (no microphone)'}")

    # Step 6: Get user name
    if not auto:
        try:
            name = input("\n[6/7] What should I call you? ").strip()
            if name:
                cfg.user_name = name
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print("\n[6/7] Skipping (auto mode)")

    # Step 7: Save config & register model
    print("\n[7/7] Saving configuration...")
    storage_path = Path(cfg.storage.path)
    storage_path.mkdir(parents=True, exist_ok=True)

    config_file = Path("homie.config.yaml")
    config_data = cfg.model_dump()
    config_file.write_text(yaml.dump(config_data, default_flow_style=False))
    print(f"  Config saved to {config_file}")
    print(f"  Data directory: {storage_path}")

    # Initialize model registry and register model
    registry = ModelRegistry(storage_path / cfg.storage.models_dir)
    registry.initialize()

    if use_cloud:
        entry = registry.register(
            name=cfg.llm.model_path,
            path=cfg.llm.model_path,
            format="cloud",
            params="cloud",
        )
        registry.set_active(cfg.llm.model_path)
        print(f"  Cloud model registered: {cfg.llm.model_path} (active)")
    elif model_path and rec:
        entry = registry.register(
            name=rec["model"],
            path=model_path,
            format=rec["format"],
            params="35B-A3B",
            repo_id=rec.get("repo_id", ""),
            quant=rec["quant"],
        )
        registry.set_active(rec["model"])
        print(f"  Model registered: {rec['model']} (active)")

    print("\n" + "=" * 50)
    print("  Setup complete! Run 'homie start' to begin.")
    print("=" * 50)

    return cfg
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_init_cloud.py -v`
Expected: Both tests PASS

**Step 5: Run all tests**

Run: `pytest tests/ -q`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/homie_app/init.py tests/unit/test_init_cloud.py
git commit -m "feat: add cloud API branch to setup wizard"
```

---

### Task 6: Update Model List Display for Cloud Entries

**Files:**
- Modify: `src/homie_app/cli.py:146-153` (cmd_model list display)

**Step 1: Update model list display**

In `src/homie_app/cli.py`, replace the model list display (line 146-153):

```python
    if args.model_command == "list":
        models = registry.list_models()
        if not models:
            print("No models installed. Run 'homie model download <repo_id>' or 'homie init'.")
        for m in models:
            active = " [ACTIVE]" if m.active else ""
            print(f"  {m.name} ({m.params}, {m.format}){active}")
            if m.format == "cloud":
                print(f"    Endpoint: cloud API")
            else:
                print(f"    Path: {m.path}")
```

**Step 2: Run all tests**

Run: `pytest tests/ -q`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/homie_app/cli.py
git commit -m "feat: display cloud model info in model list"
```

---

### Task 7: Full Integration Test

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (should be ~210+ tests now)

**Step 2: Manual smoke test — cloud setup**

Run: `homie init` and choose cloud setup with a real API key (if available), or verify the prompts display correctly and Ctrl+C exits cleanly.

**Step 3: Manual smoke test — model switch**

```bash
homie model list    # should show both local and cloud entries
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete cloud API backend integration"
```
