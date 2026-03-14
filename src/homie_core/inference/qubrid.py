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
            self._available = True  # Server responded
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
        req = Request(f"{self._base_url}/chat/completions", data=data, method="POST")
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
        req = Request(f"{self._base_url}/chat/completions", data=data, method="POST")
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
