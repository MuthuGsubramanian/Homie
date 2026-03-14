# Desktop Cross-Platform Packaging & Inference Router Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package Homie AI for Linux (.deb, .rpm, AppImage) and macOS (.dmg), add a unified inference router with Qubrid cloud fallback, build a LAN network module for device sync, and set up GitHub Actions CI/CD.

**Architecture:** Extend the existing `installer/` build system with per-platform builders. Introduce `src/homie_core/inference/` as an abstraction layer over `model/engine.py` that adds routing (local → LAN → Qubrid). Add `src/homie_core/network/` for mDNS discovery and WebSocket-based device sync. Wire everything through config and daemon.

**Tech Stack:** Python 3.11+, PyInstaller, dpkg-deb, rpmbuild, appimagetool, create-dmg, zeroconf, websockets, openai (for Qubrid), GitHub Actions

**Spec:** `docs/superpowers/specs/2026-03-14-cross-platform-packaging-design.md`

---

## Chunk 1: Inference Router & Qubrid Fallback

### Task 1: Add inference and network optional dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add new optional dependency groups**

Add after the `service` group (line 98) in `pyproject.toml`:

```toml
inference = [
    "openai>=1.30",
]
network = [
    "zeroconf>=0.131",
    "websockets>=12.0",
]
```

Update the `all` group to include them:

```toml
all = ["homie-ai[model,voice,context,storage,app,neural,email,social,messaging,screen-reader,service,inference,network]"]
```

- [ ] **Step 2: Add platform marker to screen-reader pywin32**

Change line 97:
```toml
screen-reader = ["mss>=9.0", "Pillow>=10.0", "pywin32>=306; sys_platform == 'win32'"]
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add inference and network optional deps, fix screen-reader platform marker"
```

---

### Task 2: Add inference config to HomieConfig

**Files:**
- Modify: `src/homie_core/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing test for inference config loading**

Create test in `tests/unit/test_config.py`:

```python
def test_inference_config_defaults():
    """InferenceConfig should have sensible defaults when not in YAML."""
    from homie_core.config import HomieConfig
    cfg = HomieConfig()
    assert cfg.inference.priority == ["local", "lan", "qubrid"]
    assert cfg.inference.qubrid.enabled is True
    assert cfg.inference.qubrid.model == "Qwen/Qwen3.5-Flash"
    assert cfg.inference.qubrid.base_url == "https://platform.qubrid.com/v1"
    assert cfg.inference.qubrid.timeout == 30
    assert cfg.inference.lan.prefer_desktop is True
    assert cfg.inference.lan.max_latency_ms == 500


def test_inference_config_from_yaml(tmp_path):
    """InferenceConfig should load from YAML."""
    from homie_core.config import load_config
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
inference:
  priority: [lan, local, qubrid]
  qubrid:
    enabled: false
    model: "custom/model"
    timeout: 60
  lan:
    prefer_desktop: false
    max_latency_ms: 1000
""")
    cfg = load_config(cfg_file)
    assert cfg.inference.priority == ["lan", "local", "qubrid"]
    assert cfg.inference.qubrid.enabled is False
    assert cfg.inference.qubrid.model == "custom/model"
    assert cfg.inference.qubrid.timeout == 60
    assert cfg.inference.lan.max_latency_ms == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_config.py::test_inference_config_defaults tests/unit/test_config.py::test_inference_config_from_yaml -v`
Expected: FAIL — `HomieConfig` has no `inference` field

- [ ] **Step 3: Add InferenceConfig to config.py**

Add before the `HomieConfig` class in `src/homie_core/config.py`:

```python
class QubridConfig(BaseModel):
    enabled: bool = True
    model: str = "Qwen/Qwen3.5-Flash"
    base_url: str = "https://platform.qubrid.com/v1"
    timeout: int = 30

class LANInferenceConfig(BaseModel):
    prefer_desktop: bool = True
    max_latency_ms: int = 500

class InferenceConfig(BaseModel):
    priority: list[str] = ["local", "lan", "qubrid"]
    qubrid: QubridConfig = QubridConfig()
    lan: LANInferenceConfig = LANInferenceConfig()
```

Add `inference: InferenceConfig = InferenceConfig()` to `HomieConfig`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_config.py::test_inference_config_defaults tests/unit/test_config.py::test_inference_config_from_yaml -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/config.py tests/unit/test_config.py
git commit -m "feat: add InferenceConfig with Qubrid and LAN settings"
```

---

### Task 3: Create Qubrid client

**Files:**
- Create: `src/homie_core/inference/__init__.py`
- Create: `src/homie_core/inference/qubrid.py`
- Test: `tests/unit/test_inference/__init__.py`
- Test: `tests/unit/test_inference/test_qubrid.py`

- [ ] **Step 1: Create package init**

```python
# src/homie_core/inference/__init__.py
"""Inference routing — local, LAN, and cloud backends."""
```

```python
# tests/unit/test_inference/__init__.py
```

- [ ] **Step 2: Write failing tests for QubridClient**

```python
# tests/unit/test_inference/test_qubrid.py
"""Tests for Qubrid cloud inference client."""
from unittest.mock import patch, MagicMock
import json
import pytest

from homie_core.inference.qubrid import QubridClient


def test_qubrid_client_init():
    client = QubridClient(
        api_key="test-key",
        model="Qwen/Qwen3.5-Flash",
        base_url="https://platform.qubrid.com/v1",
        timeout=30,
    )
    assert client.model == "Qwen/Qwen3.5-Flash"
    assert client.is_available is False  # Not verified yet


def test_qubrid_generate_success():
    client = QubridClient(
        api_key="test-key",
        model="Qwen/Qwen3.5-Flash",
        base_url="https://platform.qubrid.com/v1",
        timeout=30,
    )
    mock_response = {
        "choices": [{"message": {"content": "Hello!"}}]
    }
    with patch("homie_core.inference.qubrid.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        result = client.generate("Hi", max_tokens=100)
    assert result == "Hello!"


def test_qubrid_generate_timeout():
    client = QubridClient(
        api_key="test-key",
        model="Qwen/Qwen3.5-Flash",
        base_url="https://platform.qubrid.com/v1",
        timeout=1,
    )
    with patch("homie_core.inference.qubrid.urlopen") as mock_urlopen:
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("timeout")
        with pytest.raises(ConnectionError):
            client.generate("Hi")


def test_qubrid_generate_rate_limited():
    client = QubridClient(
        api_key="test-key",
        model="Qwen/Qwen3.5-Flash",
        base_url="https://platform.qubrid.com/v1",
        timeout=30,
    )
    with patch("homie_core.inference.qubrid.urlopen") as mock_urlopen:
        from urllib.error import HTTPError
        from io import BytesIO
        error = HTTPError(
            url="https://platform.qubrid.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=MagicMock(),
            fp=BytesIO(json.dumps({"error": {"message": "Rate limited"}}).encode()),
        )
        error.headers = {"Retry-After": "5"}
        mock_urlopen.side_effect = error
        with pytest.raises(RuntimeError, match="Rate limited"):
            client.generate("Hi")


def test_qubrid_check_available_success():
    client = QubridClient(
        api_key="test-key",
        model="Qwen/Qwen3.5-Flash",
        base_url="https://platform.qubrid.com/v1",
        timeout=30,
    )
    with patch("homie_core.inference.qubrid.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        assert client.check_available() is True
        assert client.is_available is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_inference/test_qubrid.py -v`
Expected: FAIL — `QubridClient` not defined

- [ ] **Step 4: Implement QubridClient**

```python
# src/homie_core/inference/qubrid.py
"""Qubrid cloud inference client — OpenAI-compatible API."""
from __future__ import annotations

import json
import logging
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class QubridClient:
    """Thin client for the Qubrid platform (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str,
        model: str = "Qwen/Qwen3.5-Flash",
        base_url: str = "https://platform.qubrid.com/v1",
        timeout: int = 30,
    ):
        self._api_key = api_key
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._available: bool = False

    @property
    def is_available(self) -> bool:
        return self._available

    def check_available(self) -> bool:
        """Ping the API to check reachability."""
        try:
            req = Request(f"{self._base_url}/models", method="GET")
            req.add_header("Authorization", f"Bearer {self._api_key}")
            with urlopen(req, timeout=10):
                pass
            self._available = True
        except HTTPError:
            # Server responded — connection works
            self._available = True
        except (URLError, OSError):
            self._available = False
        return self._available

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate a response. Raises ConnectionError or RuntimeError on failure."""
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self._base_url}/chat/completions", data=data, method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            with urlopen(req, timeout=self._timeout) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"].get("content", "") or ""
        except HTTPError as exc:
            try:
                body = json.loads(exc.read())
                msg = body.get("error", {}).get("message", str(exc))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"Qubrid API error ({exc.code}): {msg}") from exc
        except (URLError, OSError) as exc:
            raise ConnectionError(f"Cannot reach Qubrid API: {exc}") from exc

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Stream a response. Yields content chunks."""
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self._base_url}/chat/completions", data=data, method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
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
        except HTTPError as exc:
            try:
                body = json.loads(exc.read())
                msg = body.get("error", {}).get("message", str(exc))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"Qubrid API error ({exc.code}): {msg}") from exc
        except (URLError, OSError) as exc:
            raise ConnectionError(f"Cannot reach Qubrid API: {exc}") from exc
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_inference/test_qubrid.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/inference/ tests/unit/test_inference/
git commit -m "feat: add QubridClient for cloud inference fallback"
```

---

### Task 4: Create InferenceRouter

**Files:**
- Create: `src/homie_core/inference/router.py`
- Test: `tests/unit/test_inference/test_router.py`

- [ ] **Step 1: Write failing tests for InferenceRouter**

```python
# tests/unit/test_inference/test_router.py
"""Tests for the unified inference router."""
from unittest.mock import MagicMock, patch
import pytest

from homie_core.inference.router import InferenceRouter


def _make_config(priority=None):
    """Create a minimal config-like object."""
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
    """When local model is loaded, router should use it."""
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = "local response"

    router = InferenceRouter(config=_make_config(), model_engine=engine)
    result = router.generate("hello")
    assert result == "local response"
    engine.generate.assert_called_once()


def test_router_falls_back_to_qubrid():
    """When no local model, router should fall back to Qubrid."""
    engine = MagicMock()
    engine.is_loaded = False

    router = InferenceRouter(
        config=_make_config(), model_engine=engine, qubrid_api_key="test-key"
    )
    with patch.object(router, "_qubrid") as mock_qubrid:
        mock_qubrid.is_available = True
        mock_qubrid.generate.return_value = "cloud response"
        result = router.generate("hello")
    assert result == "cloud response"


def test_router_all_sources_fail():
    """When all sources fail, router should raise RuntimeError."""
    engine = MagicMock()
    engine.is_loaded = False

    router = InferenceRouter(config=_make_config(), model_engine=engine)
    # No Qubrid key, no LAN
    with pytest.raises(RuntimeError, match="All inference sources unavailable"):
        router.generate("hello")


def test_router_reports_active_source_local():
    engine = MagicMock()
    engine.is_loaded = True
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    assert router.active_source == "Local"


def test_router_reports_active_source_qubrid():
    engine = MagicMock()
    engine.is_loaded = False
    router = InferenceRouter(
        config=_make_config(), model_engine=engine, qubrid_api_key="test-key"
    )
    with patch.object(router, "_qubrid") as mock_qubrid:
        mock_qubrid.is_available = True
        assert router.active_source == "Homie Intelligence (Cloud)"


def test_router_fallback_banner_message():
    engine = MagicMock()
    engine.is_loaded = False
    router = InferenceRouter(
        config=_make_config(), model_engine=engine, qubrid_api_key="test-key"
    )
    with patch.object(router, "_qubrid") as mock_qubrid:
        mock_qubrid.is_available = True
        assert router.fallback_banner == "No local model found! Using Homie's intelligence until local model is setup!"


def test_router_no_banner_when_local():
    engine = MagicMock()
    engine.is_loaded = True
    router = InferenceRouter(config=_make_config(), model_engine=engine)
    assert router.fallback_banner is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_inference/test_router.py -v`
Expected: FAIL — `InferenceRouter` not defined

- [ ] **Step 3: Implement InferenceRouter**

```python
# src/homie_core/inference/router.py
"""Unified inference router — local → LAN → Qubrid."""
from __future__ import annotations

import logging
from typing import Iterator, Optional

from homie_core.config import HomieConfig
from homie_core.model.engine import ModelEngine

logger = logging.getLogger(__name__)

_FALLBACK_BANNER = (
    "No local model found! Using Homie's intelligence until local model is setup!"
)


class InferenceRouter:
    """Routes inference requests through the priority chain: local → LAN → Qubrid."""

    def __init__(
        self,
        config: HomieConfig,
        model_engine: ModelEngine,
        qubrid_api_key: str = "",
    ):
        self._config = config
        self._engine = model_engine
        self._qubrid = None
        self._priority = config.inference.priority

        if qubrid_api_key and config.inference.qubrid.enabled:
            from homie_core.inference.qubrid import QubridClient
            self._qubrid = QubridClient(
                api_key=qubrid_api_key,
                model=config.inference.qubrid.model,
                base_url=config.inference.qubrid.base_url,
                timeout=config.inference.qubrid.timeout,
            )
            self._qubrid.check_available()

    @property
    def active_source(self) -> str:
        """Return a human-readable label for the current inference source."""
        for source in self._priority:
            if source == "local" and self._engine.is_loaded:
                return "Local"
            if source == "lan":
                # LAN support added in Task 9
                continue
            if source == "qubrid" and self._qubrid and self._qubrid.is_available:
                return "Homie Intelligence (Cloud)"
        return "None"

    @property
    def fallback_banner(self) -> Optional[str]:
        """Return the fallback banner if not using a local model, else None."""
        if self._engine.is_loaded:
            return None
        if self._qubrid and self._qubrid.is_available:
            return _FALLBACK_BANNER
        return None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        timeout: int = 120,
    ) -> str:
        """Route a generation request through the priority chain."""
        errors: list[str] = []

        for source in self._priority:
            try:
                if source == "local" and self._engine.is_loaded:
                    return self._engine.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop, timeout=timeout,
                    )
                if source == "lan":
                    # LAN inference added in Task 9
                    continue
                if source == "qubrid" and self._qubrid and self._qubrid.is_available:
                    return self._qubrid.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
            except Exception as e:
                logger.warning("Inference source '%s' failed: %s", source, e)
                errors.append(f"{source}: {e}")
                continue

        raise RuntimeError(
            "All inference sources unavailable. "
            "Please check your connection or download a local model. "
            f"Errors: {'; '.join(errors) if errors else 'no sources configured'}"
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Route a streaming request through the priority chain."""
        errors: list[str] = []

        for source in self._priority:
            try:
                if source == "local" and self._engine.is_loaded:
                    yield from self._engine.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
                if source == "lan":
                    continue
                if source == "qubrid" and self._qubrid and self._qubrid.is_available:
                    yield from self._qubrid.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
            except Exception as e:
                logger.warning("Inference source '%s' failed: %s", source, e)
                errors.append(f"{source}: {e}")
                continue

        raise RuntimeError(
            "All inference sources unavailable. "
            f"Errors: {'; '.join(errors) if errors else 'no sources configured'}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_inference/test_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/inference/router.py tests/unit/test_inference/test_router.py
git commit -m "feat: add InferenceRouter with local/LAN/Qubrid priority chain"
```

---

### Task 5: Wire InferenceRouter into CLI and daemon

**Files:**
- Modify: `src/homie_app/cli.py`
- Modify: `src/homie_app/daemon.py`
- Modify: `homie.config.yaml`

- [ ] **Step 1: Update `_load_model_engine` in cli.py to return InferenceRouter**

In `src/homie_app/cli.py`, add a new function after `_load_model_engine`:

```python
def _create_inference_router(cfg: HomieConfig, engine: ModelEngine) -> "InferenceRouter":
    """Wrap the model engine in an InferenceRouter with Qubrid fallback."""
    from homie_core.inference.router import InferenceRouter
    from homie_core.vault.manager import VaultManager

    # Try to get Qubrid API key from vault
    qubrid_key = ""
    try:
        vault = VaultManager()
        qubrid_key = vault.get("qubrid_api_key") or ""
    except Exception:
        pass

    router = InferenceRouter(
        config=cfg,
        model_engine=engine,
        qubrid_api_key=qubrid_key,
    )

    # Show fallback banner if applicable
    banner = router.fallback_banner
    if banner:
        print(f"\n  ℹ {banner}\n")

    return router
```

- [ ] **Step 2: Add default inference section to homie.config.yaml**

Append to `homie.config.yaml`:

```yaml
# ── Inference routing ──────────────────────────────────────────────────
inference:
  priority: [local, lan, qubrid]
  qubrid:
    enabled: true
    model: "Qwen/Qwen3.5-Flash"
    # api_key stored in vault, not config
  lan:
    prefer_desktop: true
    max_latency_ms: 500
```

- [ ] **Step 3: Commit**

```bash
git add src/homie_app/cli.py src/homie_app/daemon.py homie.config.yaml
git commit -m "feat: wire InferenceRouter into CLI with Qubrid fallback banner"
```

---

## Chunk 2: Linux Packaging

### Task 6: Create unified build entry point

**Files:**
- Create: `installer/build.py`

- [ ] **Step 1: Create the unified build script**

```python
#!/usr/bin/env python3
"""Unified cross-platform build script for Homie AI.

Usage:
    python installer/build.py --target deb
    python installer/build.py --target rpm
    python installer/build.py --target appimage
    python installer/build.py --target dmg
    python installer/build.py --target msi
    python installer/build.py --target all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALLER = ROOT / "installer"


def get_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def build_target(target: str, version: str) -> None:
    if target == "msi":
        run([sys.executable, str(INSTALLER / "build_msi.py")])
    elif target == "deb":
        run([sys.executable, str(INSTALLER / "linux" / "build_deb.py")])
    elif target == "rpm":
        run([sys.executable, str(INSTALLER / "linux" / "build_rpm.py")])
    elif target == "appimage":
        run([sys.executable, str(INSTALLER / "linux" / "build_appimage.py")])
    elif target == "dmg":
        run([sys.executable, str(INSTALLER / "macos" / "build_dmg.py")])
    else:
        print(f"ERROR: Unknown target '{target}'")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Homie AI packages")
    parser.add_argument(
        "--target", required=True,
        choices=["deb", "rpm", "appimage", "dmg", "msi", "all"],
        help="Package format to build",
    )
    parser.add_argument("--skip-freeze", action="store_true")
    args = parser.parse_args()

    version = get_version()
    print(f"Building Homie AI v{version} — target: {args.target}")

    if args.target == "all":
        import platform
        os_name = platform.system()
        if os_name == "Linux":
            for t in ["deb", "rpm", "appimage"]:
                build_target(t, version)
        elif os_name == "Darwin":
            build_target("dmg", version)
        elif os_name == "Windows":
            build_target("msi", version)
    else:
        build_target(args.target, version)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add installer/build.py
git commit -m "feat: add unified cross-platform build entry point"
```

---

### Task 7: Create Linux .deb packaging

**Files:**
- Create: `installer/linux/build_deb.py`
- Create: `installer/linux/homie.desktop`
- Create: `installer/linux/homie.service`
- Create: `installer/linux/postinst`
- Create: `installer/linux/prerm`

- [ ] **Step 1: Create the desktop entry file**

```ini
# installer/linux/homie.desktop
[Desktop Entry]
Name=Homie AI
Comment=Local-first personal AI assistant
Exec=homie start
Icon=homie
Terminal=true
Type=Application
Categories=Utility;Artificial Intelligence;
Keywords=ai;assistant;local;privacy;
```

- [ ] **Step 2: Create the systemd user service**

```ini
# installer/linux/homie.service
[Unit]
Description=Homie AI Background Daemon
After=default.target

[Service]
Type=simple
ExecStart=/usr/local/bin/homie-daemon --headless
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

- [ ] **Step 3: Create post-install script**

```bash
#!/bin/bash
# installer/linux/postinst
set -e

# Create user config directory
HOMIE_DIR="$HOME/.homie"
mkdir -p "$HOMIE_DIR"

# Install systemd user service
SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_DIR"
cp /usr/local/share/homie/homie.service "$SYSTEMD_DIR/homie.service"

echo "Homie AI installed. Run 'homie' to get started."
```

- [ ] **Step 4: Create pre-remove script**

```bash
#!/bin/bash
# installer/linux/prerm
set -e

# Stop daemon if running
systemctl --user stop homie.service 2>/dev/null || true
systemctl --user disable homie.service 2>/dev/null || true

# Remove systemd service
rm -f "$HOME/.config/systemd/user/homie.service"
systemctl --user daemon-reload 2>/dev/null || true
```

- [ ] **Step 5: Create build_deb.py**

```python
#!/usr/bin/env python3
"""Build Homie AI .deb package for Ubuntu/Debian."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INSTALLER = ROOT / "installer"
LINUX = INSTALLER / "linux"
DIST = ROOT / "dist"


def get_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def check_tool(name: str) -> bool:
    return shutil.which(name) is not None


def freeze() -> Path:
    """PyInstaller freeze for Linux."""
    print("\n=== Stage 1: PyInstaller freeze ===")
    if not check_tool("pyinstaller"):
        print("ERROR: pyinstaller not found. Install with: pip install pyinstaller")
        sys.exit(1)

    spec = INSTALLER / "homie.spec"
    run(["pyinstaller", str(spec), "--noconfirm", "--distpath", str(DIST)])

    output = DIST / "homie"
    for binary in ["homie", "homie-daemon"]:
        if not (output / binary).exists():
            print(f"ERROR: {binary} not found in {output}")
            sys.exit(1)

    return output


def build_deb(frozen_dir: Path, version: str) -> Path:
    """Package frozen binaries into a .deb."""
    print("\n=== Stage 2: Build .deb ===")

    pkg_name = "homie-ai"
    arch = "amd64"
    deb_root = DIST / f"{pkg_name}_{version}_{arch}"

    # Clean previous build
    if deb_root.exists():
        shutil.rmtree(deb_root)

    # Directory structure
    bin_dir = deb_root / "usr" / "local" / "bin"
    share_dir = deb_root / "usr" / "local" / "share" / "homie"
    apps_dir = deb_root / "usr" / "share" / "applications"
    debian_dir = deb_root / "DEBIAN"

    for d in [bin_dir, share_dir, apps_dir, debian_dir]:
        d.mkdir(parents=True)

    # Copy frozen binaries
    for item in frozen_dir.iterdir():
        dest = bin_dir / item.name if item.is_file() else share_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
            # Make binaries executable
            if item.name in ("homie", "homie-daemon"):
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
        else:
            shutil.copytree(item, share_dir / item.name)

    # Copy desktop entry
    shutil.copy2(LINUX / "homie.desktop", apps_dir / "homie.desktop")

    # Copy service file to share dir (postinst moves it)
    shutil.copy2(LINUX / "homie.service", share_dir / "homie.service")

    # Control file
    (debian_dir / "control").write_text(f"""Package: {pkg_name}
Version: {version}
Section: utils
Priority: optional
Architecture: {arch}
Maintainer: MSG <muthu.g.subramanian@outlook.com>
Description: Homie AI - Local-first personal AI assistant
 Fully local, privacy-first personal AI assistant with
 background intelligence, voice interaction, and email integration.
""")

    # Post-install & pre-remove scripts
    for script_name in ["postinst", "prerm"]:
        src = LINUX / script_name
        if src.exists():
            dest = debian_dir / script_name
            shutil.copy2(src, dest)
            dest.chmod(dest.stat().st_mode | stat.S_IEXEC)

    # Build
    deb_path = DIST / f"{pkg_name}_{version}_{arch}.deb"
    run(["dpkg-deb", "--build", str(deb_root), str(deb_path)])

    if not deb_path.exists():
        print(f"ERROR: .deb not created at {deb_path}")
        sys.exit(1)

    print(f"  .deb created: {deb_path}")
    return deb_path


def main() -> None:
    version = get_version()
    print(f"Building Homie AI .deb v{version}")

    skip_freeze = "--skip-freeze" in sys.argv
    if skip_freeze:
        frozen_dir = DIST / "homie"
        if not frozen_dir.exists():
            print("ERROR: dist/homie/ not found. Run without --skip-freeze first.")
            sys.exit(1)
    else:
        frozen_dir = freeze()

    deb_path = build_deb(frozen_dir, version)
    size_mb = deb_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"  SUCCESS: {deb_path.name} ({size_mb:.1f} MB)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add installer/linux/
git commit -m "feat: add Linux .deb packaging with systemd service and desktop entry"
```

---

### Task 8: Create Linux .rpm and AppImage packaging

**Files:**
- Create: `installer/linux/build_rpm.py`
- Create: `installer/linux/build_appimage.py`
- Create: `installer/linux/homie.spec.rpmbuild`
- Create: `installer/linux/AppRun`

- [ ] **Step 1: Create RPM spec template**

```spec
# installer/linux/homie.spec.rpmbuild
Name:           homie-ai
Version:        %{version}
Release:        1%{?dist}
Summary:        Local-first personal AI assistant

License:        MPL-2.0
URL:            https://heyhomie.app

%description
Fully local, privacy-first personal AI assistant with
background intelligence, voice interaction, and email integration.

%install
mkdir -p %{buildroot}/usr/local/bin
mkdir -p %{buildroot}/usr/local/share/homie
mkdir -p %{buildroot}/usr/share/applications
cp -r %{_sourcedir}/homie/* %{buildroot}/usr/local/share/homie/
ln -sf /usr/local/share/homie/homie %{buildroot}/usr/local/bin/homie
ln -sf /usr/local/share/homie/homie-daemon %{buildroot}/usr/local/bin/homie-daemon
cp %{_sourcedir}/homie.desktop %{buildroot}/usr/share/applications/

%files
/usr/local/bin/homie
/usr/local/bin/homie-daemon
/usr/local/share/homie/
/usr/share/applications/homie.desktop

%post
mkdir -p "$HOME/.homie"
mkdir -p "$HOME/.config/systemd/user"
cp /usr/local/share/homie/homie.service "$HOME/.config/systemd/user/homie.service" 2>/dev/null || true

%preun
systemctl --user stop homie.service 2>/dev/null || true
systemctl --user disable homie.service 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/homie.service"
```

- [ ] **Step 2: Create build_rpm.py**

```python
#!/usr/bin/env python3
"""Build Homie AI .rpm package for Fedora/RHEL."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INSTALLER = ROOT / "installer"
LINUX = INSTALLER / "linux"
DIST = ROOT / "dist"


def get_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def main() -> None:
    version = get_version()
    print(f"Building Homie AI .rpm v{version}")

    frozen_dir = DIST / "homie"
    if not frozen_dir.exists():
        print("ERROR: dist/homie/ not found. Run PyInstaller freeze first.")
        print("  python installer/linux/build_deb.py  (freezes as side effect)")
        sys.exit(1)

    if not shutil.which("rpmbuild"):
        print("ERROR: rpmbuild not found. Install with: sudo dnf install rpm-build")
        sys.exit(1)

    # Set up rpmbuild tree
    rpmbuild_root = DIST / "rpmbuild"
    for d in ["BUILD", "RPMS", "SOURCES", "SPECS", "SRPMS"]:
        (rpmbuild_root / d).mkdir(parents=True, exist_ok=True)

    # Copy sources
    sources = rpmbuild_root / "SOURCES"
    if (sources / "homie").exists():
        shutil.rmtree(sources / "homie")
    shutil.copytree(frozen_dir, sources / "homie")
    shutil.copy2(LINUX / "homie.desktop", sources / "homie.desktop")
    shutil.copy2(LINUX / "homie.service", sources / "homie.service")

    # Copy and fill spec
    spec_src = LINUX / "homie.spec.rpmbuild"
    spec_content = spec_src.read_text().replace("%{version}", version)
    spec_dest = rpmbuild_root / "SPECS" / "homie-ai.spec"
    spec_dest.write_text(spec_content)

    # Build
    run([
        "rpmbuild", "-bb",
        f"--define=_topdir {rpmbuild_root}",
        f"--define=_sourcedir {sources}",
        str(spec_dest),
    ])

    # Find the built RPM
    rpm_dir = rpmbuild_root / "RPMS"
    rpms = list(rpm_dir.rglob("*.rpm"))
    if rpms:
        final = DIST / rpms[0].name
        shutil.copy2(rpms[0], final)
        size_mb = final.stat().st_size / (1024 * 1024)
        print(f"\n{'=' * 50}")
        print(f"  SUCCESS: {final.name} ({size_mb:.1f} MB)")
        print(f"{'=' * 50}")
    else:
        print("ERROR: No .rpm found after build")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create AppRun script**

```bash
#!/bin/bash
# installer/linux/AppRun
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/homie" "$@"
```

- [ ] **Step 4: Create build_appimage.py**

```python
#!/usr/bin/env python3
"""Build Homie AI AppImage (universal Linux package)."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INSTALLER = ROOT / "installer"
LINUX = INSTALLER / "linux"
DIST = ROOT / "dist"


def get_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT, env=env)


def main() -> None:
    version = get_version()
    print(f"Building Homie AI AppImage v{version}")

    frozen_dir = DIST / "homie"
    if not frozen_dir.exists():
        print("ERROR: dist/homie/ not found. Run PyInstaller freeze first.")
        sys.exit(1)

    if not shutil.which("appimagetool"):
        print("ERROR: appimagetool not found.")
        print("  Download from: https://github.com/AppImage/AppImageKit/releases")
        sys.exit(1)

    # Create AppDir structure
    appdir = DIST / "HomieAI.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)

    usr_bin = appdir / "usr" / "bin"
    usr_lib = appdir / "usr" / "lib"
    usr_bin.mkdir(parents=True)
    usr_lib.mkdir(parents=True)

    # Copy frozen binaries
    for item in frozen_dir.iterdir():
        dest = usr_bin / item.name if item.is_file() else usr_lib / item.name
        if item.is_file():
            shutil.copy2(item, dest)
            if item.name in ("homie", "homie-daemon"):
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
        else:
            shutil.copytree(item, dest)

    # AppRun
    apprun = appdir / "AppRun"
    shutil.copy2(LINUX / "AppRun", apprun)
    apprun.chmod(apprun.stat().st_mode | stat.S_IEXEC)

    # Desktop entry at root
    shutil.copy2(LINUX / "homie.desktop", appdir / "homie.desktop")

    # Placeholder icon (required by AppImage)
    icon_path = appdir / "homie.png"
    if not icon_path.exists():
        # Create a minimal 1x1 PNG as placeholder
        icon_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    # Build AppImage
    appimage_name = f"HomieAI-{version}-x86_64.AppImage"
    appimage_path = DIST / appimage_name

    env = os.environ.copy()
    env["ARCH"] = "x86_64"
    run(["appimagetool", str(appdir), str(appimage_path)], env=env)

    if not appimage_path.exists():
        print(f"ERROR: AppImage not created at {appimage_path}")
        sys.exit(1)

    size_mb = appimage_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"  SUCCESS: {appimage_name} ({size_mb:.1f} MB)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add installer/linux/
git commit -m "feat: add Linux .rpm and AppImage packaging"
```

---

## Chunk 3: macOS Packaging

### Task 9: Create macOS .dmg packaging

**Files:**
- Create: `installer/macos/build_dmg.py`
- Create: `installer/macos/com.heyhomie.daemon.plist`
- Create: `installer/macos/homie-macos.spec`
- Create: `installer/macos/uninstall.sh`

- [ ] **Step 1: Create macOS PyInstaller spec**

```python
# installer/macos/homie-macos.spec
# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Homie AI on macOS — .app bundle."""
from pathlib import Path
import os

block_cipher = None
ROOT = Path(SPECPATH).parent.parent
if not (ROOT / "src" / "homie_app" / "cli.py").exists():
    ROOT = Path(os.getcwd())

a_cli = Analysis(
    [str(ROOT / "src" / "homie_app" / "cli.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        (str(ROOT / "homie.config.yaml"), "."),
    ],
    hiddenimports=[
        "pydantic", "pydantic_core", "pydantic.deprecated",
        "pydantic.deprecated.decorator",
        "cryptography", "cryptography.hazmat", "cryptography.hazmat.primitives",
        "cffi", "_cffi_backend",
        "keyring", "keyring.backends",
        "feedparser",
        "prompt_toolkit", "prompt_toolkit.shortcuts", "prompt_toolkit.history",
        "psutil", "requests", "yaml",
        "homie_app", "homie_app.cli", "homie_app.daemon",
        "homie_app.console", "homie_app.console.console",
        "homie_app.console.router", "homie_app.console.commands",
        "homie_app.tray", "homie_app.tray.app",
        "homie_app.wizard",
        "homie_app.service",
        "homie_core", "homie_core.brain",
        "homie_core.brain.engine", "homie_core.brain.tool_registry",
        "homie_core.config",
        "homie_core.memory", "homie_core.memory.working",
        "homie_core.memory.semantic", "homie_core.memory.episodic",
        "homie_core.inference", "homie_core.inference.router",
        "homie_core.inference.qubrid",
        "homie_core.email", "homie_core.email.gmail_provider",
        "homie_core.context",
        "homie_core.hardware", "homie_core.hardware.detector",
        "homie_core.model", "homie_core.model.downloader",
        "homie_core.neural", "homie_core.neural.model_manager",
        "homie_core.notifications",
        "homie_core.voice",
        "homie_core.storage", "homie_core.storage.database",
        "homie_core.storage.vectors",
        "homie_core.rag",
        "homie_core.plugins",
        "chromadb", "chromadb.config",
        "uvicorn", "fastapi",
        "apscheduler", "apscheduler.schedulers.background",
        "numpy",
        "google.auth", "google.oauth2.credentials",
        "googleapiclient",
    ],
    hookspath=[],
    excludes=[
        "matplotlib", "scipy", "notebook", "jupyterlab",
        "pytest", "sphinx", "setuptools", "pip", "wheel",
        "tkinter", "_tkinter",
        "IPython", "ipykernel",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_daemon = Analysis(
    [str(ROOT / "src" / "homie_app" / "daemon.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[],
    hiddenimports=a_cli.hiddenimports,
    hookspath=[],
    excludes=a_cli.excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)
pyz_daemon = PYZ(a_daemon.pure, a_daemon.zipped_data, cipher=block_cipher)

exe_cli = EXE(
    pyz_cli, a_cli.scripts, [],
    exclude_binaries=True,
    name="homie",
    debug=False,
    strip=False,
    upx=True,
    console=True,
)

exe_daemon = EXE(
    pyz_daemon, a_daemon.scripts, [],
    exclude_binaries=True,
    name="homie-daemon",
    debug=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe_cli, a_cli.binaries, a_cli.zipfiles, a_cli.datas,
    exe_daemon, a_daemon.binaries, a_daemon.zipfiles, a_daemon.datas,
    strip=False, upx=True, name="homie",
)

app = BUNDLE(
    coll,
    name="Homie AI.app",
    bundle_identifier="com.heyhomie.app",
    info_plist={
        "CFBundleName": "Homie AI",
        "CFBundleDisplayName": "Homie AI",
        "CFBundleVersion": "0.2.0",
        "CFBundleShortVersionString": "0.2.0",
        "LSMinimumSystemVersion": "10.15",
        "NSHighResolutionCapable": True,
    },
)
```

- [ ] **Step 2: Create LaunchAgent plist**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.heyhomie.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Applications/Homie AI.app/Contents/MacOS/homie-daemon</string>
        <string>--headless</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/homie-daemon.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/homie-daemon.log</string>
</dict>
</plist>
```

- [ ] **Step 3: Create uninstall script**

```bash
#!/bin/bash
# installer/macos/uninstall.sh
# Uninstall Homie AI from macOS

echo "Uninstalling Homie AI..."

# Stop and unload daemon
launchctl unload ~/Library/LaunchAgents/com.heyhomie.daemon.plist 2>/dev/null
rm -f ~/Library/LaunchAgents/com.heyhomie.daemon.plist

# Remove app
if [ -d "/Applications/Homie AI.app" ]; then
    osascript -e 'tell application "Finder" to delete POSIX file "/Applications/Homie AI.app"' 2>/dev/null || rm -rf "/Applications/Homie AI.app"
fi

echo "Homie AI uninstalled. Your data in ~/.homie/ was preserved."
echo "To remove all data: rm -rf ~/.homie/"
```

- [ ] **Step 4: Create build_dmg.py**

```python
#!/usr/bin/env python3
"""Build Homie AI .dmg for macOS."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INSTALLER = ROOT / "installer"
MACOS = INSTALLER / "macos"
DIST = ROOT / "dist"


def get_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def main() -> None:
    version = get_version()
    print(f"Building Homie AI .dmg v{version}")

    # Stage 1: PyInstaller
    skip_freeze = "--skip-freeze" in sys.argv
    if skip_freeze:
        app_path = DIST / "Homie AI.app"
    else:
        print("\n=== Stage 1: PyInstaller freeze ===")
        if not shutil.which("pyinstaller"):
            print("ERROR: pyinstaller not found")
            sys.exit(1)
        spec = MACOS / "homie-macos.spec"
        run(["pyinstaller", str(spec), "--noconfirm", "--distpath", str(DIST)])
        app_path = DIST / "Homie AI.app"

    if not app_path.exists():
        print(f"ERROR: {app_path} not found")
        sys.exit(1)

    # Copy uninstall script into .app
    resources = app_path / "Contents" / "Resources"
    resources.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MACOS / "uninstall.sh", resources / "uninstall.sh")

    # Copy LaunchAgent plist into .app
    shutil.copy2(
        MACOS / "com.heyhomie.daemon.plist",
        resources / "com.heyhomie.daemon.plist",
    )

    # Stage 2: Create DMG
    print("\n=== Stage 2: Create .dmg ===")
    dmg_name = f"HomieAI-{version}.dmg"
    dmg_path = DIST / dmg_name

    if dmg_path.exists():
        dmg_path.unlink()

    if shutil.which("create-dmg"):
        run([
            "create-dmg",
            "--volname", "Homie AI",
            "--window-size", "600", "400",
            "--app-drop-link", "400", "200",
            "--icon", "Homie AI.app", "200", "200",
            str(dmg_path),
            str(app_path),
        ])
    else:
        # Fallback: hdiutil
        run([
            "hdiutil", "create",
            "-volname", "Homie AI",
            "-srcfolder", str(app_path),
            "-ov",
            "-format", "UDZO",
            str(dmg_path),
        ])

    if not dmg_path.exists():
        print(f"ERROR: .dmg not created at {dmg_path}")
        sys.exit(1)

    size_mb = dmg_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"  SUCCESS: {dmg_name} ({size_mb:.1f} MB)")
    print(f"  NOTE: This .dmg is unsigned. Users must right-click → Open")
    print(f"        to bypass Gatekeeper on first launch.")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add installer/macos/
git commit -m "feat: add macOS .dmg packaging with LaunchAgent and uninstall script"
```

---

## Chunk 4: LAN Network Module

### Task 10: Create network protocol definitions

**Files:**
- Create: `src/homie_core/network/__init__.py`
- Create: `src/homie_core/network/protocol.py`
- Test: `tests/unit/test_network/__init__.py`
- Test: `tests/unit/test_network/test_protocol.py`

- [ ] **Step 1: Write failing tests for protocol messages**

```python
# tests/unit/test_network/test_protocol.py
"""Tests for HomieSync protocol message types."""
import json
import pytest
from homie_core.network.protocol import (
    ProtocolMessage, MessageType, InferenceRequest, InferenceResponse,
    StatusMessage, MemorySyncMessage, HelloMessage, PROTOCOL_VERSION,
)


def test_hello_message_serialization():
    msg = HelloMessage(device_id="desktop-1", device_name="My PC", protocol_version=PROTOCOL_VERSION)
    data = msg.to_json()
    parsed = json.loads(data)
    assert parsed["type"] == "hello"
    assert parsed["protocol_version"] == PROTOCOL_VERSION
    assert parsed["payload"]["device_id"] == "desktop-1"


def test_inference_request_roundtrip():
    req = InferenceRequest(
        request_id="req-001",
        prompt="Hello",
        max_tokens=100,
        temperature=0.7,
    )
    data = req.to_json()
    parsed = ProtocolMessage.from_json(data)
    assert isinstance(parsed, InferenceRequest)
    assert parsed.prompt == "Hello"
    assert parsed.request_id == "req-001"


def test_inference_response_roundtrip():
    resp = InferenceResponse(
        request_id="req-001",
        content="Hi there!",
        source="local",
    )
    data = resp.to_json()
    parsed = ProtocolMessage.from_json(data)
    assert isinstance(parsed, InferenceResponse)
    assert parsed.content == "Hi there!"


def test_status_message():
    msg = StatusMessage(
        device_id="phone-1",
        model_loaded=True,
        model_name="Qwen-1.5B",
        daemon_running=True,
        battery_level=85,
    )
    data = msg.to_json()
    parsed = ProtocolMessage.from_json(data)
    assert isinstance(parsed, StatusMessage)
    assert parsed.battery_level == 85


def test_unknown_message_type():
    data = json.dumps({"type": "unknown_type", "protocol_version": "1.0.0", "payload": {}})
    with pytest.raises(ValueError, match="Unknown message type"):
        ProtocolMessage.from_json(data)


def test_major_version_mismatch():
    data = json.dumps({"type": "hello", "protocol_version": "99.0.0", "payload": {"device_id": "x", "device_name": "y"}})
    with pytest.raises(ValueError, match="Incompatible protocol version"):
        ProtocolMessage.from_json(data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_network/test_protocol.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement protocol.py**

```python
# src/homie_core/network/__init__.py
"""Network module — LAN discovery, sync, and protocol."""
```

```python
# tests/unit/test_network/__init__.py
```

```python
# src/homie_core/network/protocol.py
"""HomieSync protocol — message types for LAN device communication."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

PROTOCOL_VERSION = "1.0.0"


def _major(version: str) -> int:
    return int(version.split(".")[0])


@dataclass
class ProtocolMessage:
    """Base class for all protocol messages."""

    type: str = ""

    def to_json(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "type"}
        return json.dumps({
            "type": self.type,
            "protocol_version": PROTOCOL_VERSION,
            "payload": payload,
        })

    @staticmethod
    def from_json(data: str) -> "ProtocolMessage":
        obj = json.loads(data)
        version = obj.get("protocol_version", "1.0.0")
        if _major(version) != _major(PROTOCOL_VERSION):
            raise ValueError(
                f"Incompatible protocol version: {version} (expected {PROTOCOL_VERSION} major)"
            )
        msg_type = obj["type"]
        payload = obj.get("payload", {})

        registry = {
            "hello": HelloMessage,
            "inference_request": InferenceRequest,
            "inference_response": InferenceResponse,
            "status": StatusMessage,
            "memory_sync": MemorySyncMessage,
            "command": CommandMessage,
            "command_result": CommandResultMessage,
            "unpair": UnpairMessage,
        }
        cls = registry.get(msg_type)
        if cls is None:
            raise ValueError(f"Unknown message type: {msg_type}")
        return cls(**payload)


@dataclass
class HelloMessage(ProtocolMessage):
    device_id: str = ""
    device_name: str = ""
    protocol_version: str = PROTOCOL_VERSION
    type: str = field(default="hello", init=False)


@dataclass
class InferenceRequest(ProtocolMessage):
    request_id: str = ""
    prompt: str = ""
    max_tokens: int = 1024
    temperature: float = 0.7
    stop: Optional[list[str]] = None
    type: str = field(default="inference_request", init=False)


@dataclass
class InferenceResponse(ProtocolMessage):
    request_id: str = ""
    content: str = ""
    source: str = ""
    error: str = ""
    type: str = field(default="inference_response", init=False)


@dataclass
class StatusMessage(ProtocolMessage):
    device_id: str = ""
    model_loaded: bool = False
    model_name: str = ""
    daemon_running: bool = False
    battery_level: Optional[int] = None
    type: str = field(default="status", init=False)


@dataclass
class MemorySyncMessage(ProtocolMessage):
    device_id: str = ""
    sync_type: str = ""  # "full" or "incremental"
    data_type: str = ""  # "conversation", "episodic", "semantic", "settings"
    entries: list = field(default_factory=list)
    sync_version: int = 0
    type: str = field(default="memory_sync", init=False)


@dataclass
class CommandMessage(ProtocolMessage):
    command_id: str = ""
    command: str = ""
    args: dict = field(default_factory=dict)
    type: str = field(default="command", init=False)


@dataclass
class CommandResultMessage(ProtocolMessage):
    command_id: str = ""
    result: str = ""
    success: bool = True
    type: str = field(default="command_result", init=False)


@dataclass
class UnpairMessage(ProtocolMessage):
    device_id: str = ""
    type: str = field(default="unpair", init=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_network/test_protocol.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/network/ tests/unit/test_network/
git commit -m "feat: add HomieSync protocol message types with version negotiation"
```

---

### Task 11: Create mDNS discovery service

**Files:**
- Create: `src/homie_core/network/discovery.py`
- Test: `tests/unit/test_network/test_discovery.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_network/test_discovery.py
"""Tests for mDNS discovery service."""
from unittest.mock import patch, MagicMock
import pytest

from homie_core.network.discovery import HomieDiscovery


def test_discovery_init():
    discovery = HomieDiscovery(device_id="desktop-1", device_name="My PC", port=8765)
    assert discovery.device_id == "desktop-1"
    assert discovery.port == 8765
    assert discovery.is_advertising is False


def test_discovery_start_advertising():
    with patch("homie_core.network.discovery.Zeroconf") as MockZC:
        mock_zc = MagicMock()
        MockZC.return_value = mock_zc

        discovery = HomieDiscovery(device_id="desktop-1", device_name="My PC", port=8765)
        discovery.start_advertising()

        assert discovery.is_advertising is True
        mock_zc.register_service.assert_called_once()


def test_discovery_stop_advertising():
    with patch("homie_core.network.discovery.Zeroconf") as MockZC:
        mock_zc = MagicMock()
        MockZC.return_value = mock_zc

        discovery = HomieDiscovery(device_id="desktop-1", device_name="My PC", port=8765)
        discovery.start_advertising()
        discovery.stop_advertising()

        assert discovery.is_advertising is False
        mock_zc.unregister_service.assert_called_once()
        mock_zc.close.assert_called_once()


def test_discovery_browse_devices():
    with patch("homie_core.network.discovery.Zeroconf"):
        with patch("homie_core.network.discovery.ServiceBrowser"):
            discovery = HomieDiscovery(device_id="desktop-1", device_name="My PC", port=8765)
            # Initially no devices
            assert discovery.discovered_devices == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_network/test_discovery.py -v`
Expected: FAIL

- [ ] **Step 3: Implement discovery.py**

```python
# src/homie_core/network/discovery.py
"""mDNS/DNS-SD service advertisement and discovery for Homie LAN sync."""
from __future__ import annotations

import logging
import socket
from typing import Optional

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_homie._tcp.local."

try:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceInfo, ServiceStateChange
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False
    Zeroconf = None
    ServiceBrowser = None
    ServiceInfo = None


class HomieDiscovery:
    """Advertise and discover Homie instances on the LAN via mDNS."""

    def __init__(self, device_id: str, device_name: str, port: int = 8765):
        self.device_id = device_id
        self.device_name = device_name
        self.port = port
        self._zeroconf: Optional[Zeroconf] = None
        self._browser: Optional[ServiceBrowser] = None
        self._service_info: Optional[ServiceInfo] = None
        self._advertising = False
        self._discovered: dict[str, dict] = {}

    @property
    def is_advertising(self) -> bool:
        return self._advertising

    @property
    def discovered_devices(self) -> dict[str, dict]:
        return dict(self._discovered)

    def start_advertising(self) -> None:
        """Start advertising this Homie instance on the LAN."""
        if not HAS_ZEROCONF:
            logger.warning("zeroconf not installed — LAN discovery disabled")
            return

        self._zeroconf = Zeroconf()
        hostname = socket.gethostname()

        self._service_info = ServiceInfo(
            SERVICE_TYPE,
            f"homie-{self.device_id}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(self._get_local_ip())],
            port=self.port,
            properties={
                "device_id": self.device_id,
                "device_name": self.device_name,
                "version": "1.0.0",
            },
            server=f"{hostname}.local.",
        )
        self._zeroconf.register_service(self._service_info)
        self._advertising = True
        logger.info("Advertising Homie on LAN: %s:%d", hostname, self.port)

    def stop_advertising(self) -> None:
        """Stop advertising and close zeroconf."""
        if self._zeroconf and self._service_info:
            self._zeroconf.unregister_service(self._service_info)
            self._zeroconf.close()
        self._advertising = False
        self._zeroconf = None
        self._service_info = None

    def start_browsing(self) -> None:
        """Start discovering other Homie instances on the LAN."""
        if not HAS_ZEROCONF:
            return
        if not self._zeroconf:
            self._zeroconf = Zeroconf()
        self._browser = ServiceBrowser(
            self._zeroconf, SERVICE_TYPE, handlers=[self._on_service_state_change]
        )

    def stop_browsing(self) -> None:
        """Stop browsing for devices."""
        if self._browser:
            self._browser.cancel()
            self._browser = None

    def _on_service_state_change(
        self, zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        if state_change == ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                device_id = info.properties.get(b"device_id", b"").decode()
                if device_id and device_id != self.device_id:
                    self._discovered[device_id] = {
                        "name": info.properties.get(b"device_name", b"").decode(),
                        "host": socket.inet_ntoa(info.addresses[0]) if info.addresses else "",
                        "port": info.port,
                    }
                    logger.info("Discovered Homie device: %s at %s:%d",
                                device_id, self._discovered[device_id]["host"], info.port)
        elif state_change == ServiceStateChange.Removed:
            # Try to extract device_id from service name
            for did in list(self._discovered.keys()):
                if did in name:
                    del self._discovered[did]
                    logger.info("Device removed: %s", did)

    @staticmethod
    def _get_local_ip() -> str:
        """Get the local IP address for LAN advertisement."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_network/test_discovery.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/network/discovery.py tests/unit/test_network/test_discovery.py
git commit -m "feat: add mDNS discovery for LAN device sync"
```

---

### Task 12: Create WebSocket sync server

**Files:**
- Create: `src/homie_core/network/server.py`
- Test: `tests/unit/test_network/test_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_network/test_server.py
"""Tests for the WebSocket sync server."""
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from homie_core.network.server import SyncServer
from homie_core.network.protocol import PROTOCOL_VERSION


@pytest.fixture
def server():
    return SyncServer(device_id="desktop-1", device_name="My PC", port=8765)


def test_server_init(server):
    assert server.device_id == "desktop-1"
    assert server.port == 8765
    assert server.paired_devices == {}


def test_server_generate_pairing_code(server):
    code = server.generate_pairing_code()
    assert len(code) == 6
    assert code.isdigit()


def test_server_verify_pairing_code(server):
    code = server.generate_pairing_code()
    assert server.verify_pairing_code(code) is True
    assert server.verify_pairing_code("000000") is False


def test_server_add_paired_device(server):
    server.add_paired_device("phone-1", "My Phone", "fake-public-key")
    assert "phone-1" in server.paired_devices
    assert server.paired_devices["phone-1"]["name"] == "My Phone"


def test_server_remove_paired_device(server):
    server.add_paired_device("phone-1", "My Phone", "fake-public-key")
    server.remove_paired_device("phone-1")
    assert "phone-1" not in server.paired_devices
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_network/test_server.py -v`
Expected: FAIL

- [ ] **Step 3: Implement server.py**

```python
# src/homie_core/network/server.py
"""WebSocket sync server for Homie LAN communication."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import string
from pathlib import Path
from typing import Optional

from homie_core.network.protocol import (
    ProtocolMessage, HelloMessage, InferenceRequest, InferenceResponse,
    StatusMessage, UnpairMessage, PROTOCOL_VERSION,
)

logger = logging.getLogger(__name__)

PAIRED_DEVICES_FILE = Path.home() / ".homie" / "paired_devices.json"


class SyncServer:
    """WebSocket server for LAN device sync."""

    def __init__(self, device_id: str, device_name: str, port: int = 8765):
        self.device_id = device_id
        self.device_name = device_name
        self.port = port
        self._pairing_code: str = ""
        self._paired_devices: dict[str, dict] = {}
        self._connected_clients: dict[str, object] = {}
        self._inference_handler = None
        self._load_paired_devices()

    @property
    def paired_devices(self) -> dict[str, dict]:
        return dict(self._paired_devices)

    def set_inference_handler(self, handler) -> None:
        """Set the inference router for handling remote inference requests."""
        self._inference_handler = handler

    def generate_pairing_code(self) -> str:
        """Generate a 6-digit pairing code."""
        self._pairing_code = "".join(random.choices(string.digits, k=6))
        return self._pairing_code

    def verify_pairing_code(self, code: str) -> bool:
        """Verify a pairing code."""
        return code == self._pairing_code and self._pairing_code != ""

    def add_paired_device(self, device_id: str, name: str, public_key: str) -> None:
        """Add a device to the paired list."""
        self._paired_devices[device_id] = {
            "name": name,
            "public_key": public_key,
        }
        self._save_paired_devices()

    def remove_paired_device(self, device_id: str) -> None:
        """Remove a device from the paired list (unpair)."""
        self._paired_devices.pop(device_id, None)
        self._save_paired_devices()

    def _load_paired_devices(self) -> None:
        if PAIRED_DEVICES_FILE.exists():
            try:
                self._paired_devices = json.loads(PAIRED_DEVICES_FILE.read_text())
            except Exception:
                self._paired_devices = {}

    def _save_paired_devices(self) -> None:
        PAIRED_DEVICES_FILE.parent.mkdir(parents=True, exist_ok=True)
        PAIRED_DEVICES_FILE.write_text(json.dumps(self._paired_devices, indent=2))

    async def handle_connection(self, websocket) -> None:
        """Handle an incoming WebSocket connection."""
        device_id = None
        try:
            # Expect hello message first
            raw = await asyncio.wait_for(websocket.recv(), timeout=10)
            msg = ProtocolMessage.from_json(raw)

            if not isinstance(msg, HelloMessage):
                await websocket.close(1002, "Expected hello message")
                return

            device_id = msg.device_id
            if device_id not in self._paired_devices:
                await websocket.close(1008, "Device not paired")
                return

            self._connected_clients[device_id] = websocket
            logger.info("Device connected: %s", device_id)

            # Send our hello back
            hello = HelloMessage(device_id=self.device_id, device_name=self.device_name)
            await websocket.send(hello.to_json())

            # Message loop
            async for raw in websocket:
                try:
                    msg = ProtocolMessage.from_json(raw)
                    await self._handle_message(msg, websocket)
                except Exception as e:
                    logger.error("Error handling message: %s", e)

        except Exception as e:
            logger.error("Connection error: %s", e)
        finally:
            if device_id:
                self._connected_clients.pop(device_id, None)
                logger.info("Device disconnected: %s", device_id)

    async def _handle_message(self, msg: ProtocolMessage, websocket) -> None:
        """Route incoming messages to handlers."""
        if isinstance(msg, InferenceRequest):
            await self._handle_inference(msg, websocket)
        elif isinstance(msg, StatusMessage):
            logger.info("Status from %s: model=%s, battery=%s",
                        msg.device_id, msg.model_name, msg.battery_level)
        elif isinstance(msg, UnpairMessage):
            self.remove_paired_device(msg.device_id)
            await websocket.close(1000, "Unpaired")

    async def _handle_inference(self, req: InferenceRequest, websocket) -> None:
        """Handle a remote inference request."""
        if not self._inference_handler:
            resp = InferenceResponse(
                request_id=req.request_id,
                error="No inference handler available",
            )
        else:
            try:
                result = self._inference_handler.generate(
                    req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    stop=req.stop,
                )
                resp = InferenceResponse(
                    request_id=req.request_id,
                    content=result,
                    source="lan",
                )
            except Exception as e:
                resp = InferenceResponse(
                    request_id=req.request_id,
                    error=str(e),
                )
        await websocket.send(resp.to_json())

    async def start(self) -> None:
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed — LAN sync server disabled")
            return

        logger.info("Starting sync server on port %d", self.port)
        async with websockets.serve(self.handle_connection, "0.0.0.0", self.port):
            await asyncio.Future()  # Run forever
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_network/test_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/network/server.py tests/unit/test_network/test_server.py
git commit -m "feat: add WebSocket sync server with pairing and inference offloading"
```

---

## Chunk 5: CI/CD & Integration

### Task 13: Create GitHub Actions release workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Create the workflow**

```yaml
# .github/workflows/release.yml
name: Build & Release

on:
  push:
    tags: ["v*"]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install -e ".[all,dev]"
          pip install pyinstaller
      - name: Run tests
        run: pytest tests/unit/ -v --tb=short
      - name: Build .deb
        run: python installer/linux/build_deb.py
      - name: Build AppImage
        run: |
          wget -q https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage -O /usr/local/bin/appimagetool
          chmod +x /usr/local/bin/appimagetool
          python installer/linux/build_appimage.py
        continue-on-error: true
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: linux-packages
          path: |
            dist/*.deb
            dist/*.AppImage

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install -e ".[all,dev]"
          pip install pyinstaller
      - name: Run tests
        run: pytest tests/unit/ -v --tb=short
      - name: Build .dmg
        run: |
          brew install create-dmg || true
          python installer/macos/build_dmg.py
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: macos-packages
          path: dist/*.dmg

  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install -e ".[all,dev]"
          pip install pyinstaller
      - name: Run tests
        run: pytest tests/unit/ -v --tb=short
      - name: Build MSI
        run: python installer/build_msi.py
        continue-on-error: true
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: windows-packages
          path: dist/*.msi

  release:
    needs: [build-linux, build-macos, build-windows]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - uses: actions/download-artifact@v4
        with:
          merge-multiple: true
          path: packages/
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: packages/*
          generate_release_notes: true
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add GitHub Actions matrix build for Linux, macOS, and Windows"
```

---

### Task 14: Wire network module into daemon

**Files:**
- Modify: `src/homie_app/daemon.py`

- [ ] **Step 1: Add network initialization to HomieDaemon**

Add to the daemon's initialization (after existing service setup), inside the `HomieDaemon.start()` method:

```python
# --- LAN Network (optional) ---
try:
    from homie_core.network.discovery import HomieDiscovery
    from homie_core.network.server import SyncServer
    import uuid

    device_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, socket.gethostname()))[:8]
    device_name = socket.gethostname()

    self._discovery = HomieDiscovery(
        device_id=device_id,
        device_name=device_name,
        port=8765,
    )
    self._discovery.start_advertising()
    self._discovery.start_browsing()

    self._sync_server = SyncServer(
        device_id=device_id,
        device_name=device_name,
        port=8765,
    )
    # Wire inference router if available
    if hasattr(self, '_inference_router'):
        self._sync_server.set_inference_handler(self._inference_router)

    # Start sync server in background thread
    import threading
    def _run_sync():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._sync_server.start())

    self._sync_thread = threading.Thread(target=_run_sync, daemon=True)
    self._sync_thread.start()
    logger.info("LAN sync server started on port 8765")
except ImportError:
    logger.debug("Network module not available — LAN sync disabled")
except Exception as e:
    logger.warning("Failed to start LAN sync: %s", e)
```

- [ ] **Step 2: Add cleanup on daemon stop**

In the daemon's shutdown/cleanup section:

```python
# Stop network services
if hasattr(self, '_discovery'):
    self._discovery.stop_advertising()
    self._discovery.stop_browsing()
```

- [ ] **Step 3: Commit**

```bash
git add src/homie_app/daemon.py
git commit -m "feat: wire LAN discovery and sync server into daemon lifecycle"
```

---

### Task 15: Update Windows PyInstaller spec with new modules

**Files:**
- Modify: `installer/homie.spec`

- [ ] **Step 1: Add inference and network to hiddenimports**

Add to the `hiddenimports` list in `installer/homie.spec`:

```python
        # Inference routing
        "homie_core.inference", "homie_core.inference.router",
        "homie_core.inference.qubrid",
        # Network / LAN sync
        "homie_core.network", "homie_core.network.protocol",
        "homie_core.network.discovery", "homie_core.network.server",
        "zeroconf",
        "websockets",
```

- [ ] **Step 2: Commit**

```bash
git add installer/homie.spec
git commit -m "build: add inference and network modules to PyInstaller hiddenimports"
```

---

### Task 16: Add network config section

**Files:**
- Modify: `src/homie_core/config.py`
- Modify: `homie.config.yaml`

- [ ] **Step 1: Add NetworkConfig to config.py**

Add before `HomieConfig` class:

```python
class NetworkConfig(BaseModel):
    enabled: bool = True
    sync_port: int = 8765
    auto_discover: bool = True
    sync_scope: str = "all"  # "all", "conversations", "manual"
```

Add `network: NetworkConfig = NetworkConfig()` to `HomieConfig`.

- [ ] **Step 2: Add to homie.config.yaml**

```yaml
# ── LAN Network ────────────────────────────────────────────────────────
network:
  enabled: true
  sync_port: 8765
  auto_discover: true
  sync_scope: all  # all, conversations, manual
```

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/config.py homie.config.yaml
git commit -m "feat: add NetworkConfig for LAN sync settings"
```

---

### Task 17: Run full test suite and verify

- [ ] **Step 1: Run all unit tests**

Run: `python -m pytest tests/unit/ -v --tb=short`
Expected: All existing tests pass, plus new tests for inference and network

- [ ] **Step 2: Run specific new test modules**

Run: `python -m pytest tests/unit/test_inference/ tests/unit/test_network/ -v`
Expected: All PASS

- [ ] **Step 3: Verify build script syntax**

Run: `python -c "import ast; ast.parse(open('installer/build.py').read()); print('OK')"`
Run: `python -c "import ast; ast.parse(open('installer/linux/build_deb.py').read()); print('OK')"`
Run: `python -c "import ast; ast.parse(open('installer/linux/build_rpm.py').read()); print('OK')"`
Run: `python -c "import ast; ast.parse(open('installer/linux/build_appimage.py').read()); print('OK')"`
Run: `python -c "import ast; ast.parse(open('installer/macos/build_dmg.py').read()); print('OK')"`
Expected: All print OK

- [ ] **Step 4: Final commit with any fixes**

```bash
git add -A
git commit -m "fix: address any test failures from integration"
```
