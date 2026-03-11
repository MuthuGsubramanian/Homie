"""Backend for Hugging Face Inference API.

Uses HF's serverless inference endpoints for both chat completion and
text embeddings. Requires HF_KEY environment variable.

Supports:
- Chat completion via /models/{model}/v1/chat/completions (OpenAI-compatible)
- Streaming via SSE
- Text embeddings via /pipeline/feature-extraction/{model}
"""
from __future__ import annotations

import json
import os
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# HF Inference API base
_HF_INFERENCE_URL = "https://router.huggingface.co"
_HF_EMBEDDING_URL = "https://router.huggingface.co/pipeline/feature-extraction"

# Default models — best open-source options on HF
_DEFAULT_CHAT_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_hf_key() -> str:
    """Get HF API key from environment."""
    key = os.environ.get("HF_KEY", "")
    if not key:
        key = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if not key:
        key = os.environ.get("HF_TOKEN", "")
    return key


class HFBackend:
    """Backend for Hugging Face Inference API — chat completions."""

    def __init__(self):
        self._model: str = ""
        self._api_key: str = ""
        self._connected: bool = False

    def load(
        self,
        model_id: str = "",
        api_key: str = "",
        base_url: str = "",  # ignored, kept for interface compat
    ) -> None:
        """Connect to HF Inference API."""
        self._model = model_id or _DEFAULT_CHAT_MODEL
        self._api_key = api_key or _get_hf_key()

        if not self._api_key:
            raise ConnectionError(
                "No HF API key found. Set the HF_KEY environment variable.\n"
                "  Get your key at: https://huggingface.co/settings/tokens"
            )

        # Verify connectivity
        try:
            req = Request(
                f"{_HF_INFERENCE_URL}/models/{self._model}",
                method="GET",
            )
            req.add_header("Authorization", f"Bearer {self._api_key}")
            with urlopen(req, timeout=10):
                pass
        except HTTPError:
            pass  # 4xx means server is reachable
        except (URLError, OSError) as exc:
            raise ConnectionError(
                f"Cannot reach HF Inference API: {exc}"
            ) from exc

        self._connected = True

    def unload(self) -> None:
        self._model = ""
        self._api_key = ""
        self._connected = False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate a response using HF chat completions endpoint."""
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
        req = Request(
            f"{_HF_INFERENCE_URL}/models/{self._model}/v1/chat/completions",
            data=data,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            with urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"].get("content", "") or ""
        except HTTPError as exc:
            try:
                body = json.loads(exc.read())
                msg = body.get("error", str(exc))
                if isinstance(msg, dict):
                    msg = msg.get("message", str(msg))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"HF API error ({exc.code}): {msg}") from exc

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Stream tokens from HF chat completions endpoint."""
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
        req = Request(
            f"{_HF_INFERENCE_URL}/models/{self._model}/v1/chat/completions",
            data=data,
            method="POST",
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
                msg = body.get("error", str(exc))
                if isinstance(msg, dict):
                    msg = msg.get("message", str(msg))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"HF API error ({exc.code}): {msg}") from exc


class HFEmbeddings:
    """Lightweight embedding provider using HF Inference API.

    No ONNX, no local model download — just HTTP calls to HF.
    Provides the `embed_fn: Callable[[str], list[float]]` that all
    neural components need.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_EMBEDDING_MODEL,
        api_key: str = "",
    ):
        self._model = model_id
        self._api_key = api_key or _get_hf_key()
        self._dimension: int = 0
        self._connected = False

    def connect(self) -> None:
        """Verify API connectivity and determine embedding dimension."""
        if not self._api_key:
            raise ConnectionError(
                "No HF API key. Set the HF_KEY environment variable."
            )

        # Do a test embedding to get dimension
        try:
            test = self._call_api("test")
            self._dimension = len(test)
            self._connected = True
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to HF embeddings: {exc}"
            ) from exc

    def _call_api(self, text: str) -> list[float]:
        """Call HF feature-extraction pipeline."""
        data = json.dumps({"inputs": text}).encode("utf-8")
        req = Request(
            f"{_HF_EMBEDDING_URL}/{self._model}",
            data=data,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")

        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        # HF returns different shapes depending on model:
        # - [384] (already pooled)
        # - [[384]] (batch of 1)
        # - [[[384]]] (token-level, needs pooling)
        if isinstance(result, list):
            if result and isinstance(result[0], list):
                if result[0] and isinstance(result[0][0], list):
                    # Token-level: mean pool
                    tokens = result[0]
                    dim = len(tokens[0])
                    pooled = [0.0] * dim
                    for token in tokens:
                        for i in range(dim):
                            pooled[i] += token[i]
                    n = len(tokens)
                    return [v / n for v in pooled]
                else:
                    # Batch of 1
                    return result[0]
            else:
                # Already pooled
                return result

        raise ValueError(f"Unexpected embedding response shape: {type(result)}")

    def embed(self, text: str) -> list[float]:
        """Embed a single text. This is the `embed_fn` for neural components."""
        if not self._connected:
            self.connect()
        try:
            return self._call_api(text)
        except HTTPError as exc:
            # On rate limit or transient error, return zero vector
            if exc.code == 429:
                import time
                time.sleep(1)
                return self._call_api(text)
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_connected(self) -> bool:
        return self._connected


def discover_hf_models(api_key: str = "") -> list[dict]:
    """Discover available chat models on HF Inference API."""
    key = api_key or _get_hf_key()
    if not key:
        return []

    # Query HF for popular inference-ready models
    popular = [
        {"id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "name": "Mistral Small 3.1 24B", "params": "24B"},
        {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral 7B Instruct", "params": "7B"},
        {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B", "params": "8B"},
        {"id": "meta-llama/Llama-3.1-70B-Instruct", "name": "Llama 3.1 70B", "params": "70B"},
        {"id": "Qwen/Qwen2.5-72B-Instruct", "name": "Qwen 2.5 72B", "params": "72B"},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5 7B", "params": "7B"},
        {"id": "google/gemma-2-9b-it", "name": "Gemma 2 9B", "params": "9B"},
        {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi 3 Mini", "params": "3.8B"},
    ]
    return popular
