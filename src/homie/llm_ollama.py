from __future__ import annotations

import json
import logging
from typing import Any, Dict

import requests

from homie.config import HomieConfig, cfg_get


class LLMError(Exception):
    """Raised when Ollama generation fails."""


def _build_payload(cfg: HomieConfig, prompt: str) -> Dict[str, Any]:
    model = cfg_get(cfg, "llm", "model", default="glm-4.7-flash")
    temperature = cfg_get(cfg, "llm", "temperature", default=0.2)
    max_tokens = cfg_get(cfg, "llm", "max_tokens", default=800)

    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }


def ollama_generate(cfg: HomieConfig, prompt: str) -> str:
    base_url = cfg_get(cfg, "llm", "base_url", default="http://127.0.0.1:11434").rstrip(
        "/"
    )
    timeout = cfg_get(cfg, "llm", "timeout_sec", default=60)
    url = f"{base_url}/api/generate"
    payload = _build_payload(cfg, prompt)

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LLMError(f"Ollama request failed: {exc}") from exc

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise LLMError("Invalid JSON from Ollama response") from exc

    if "response" not in data:
        raise LLMError("Ollama response missing 'response' field")

    logging.debug("LLM raw response: %s", data.get("response", "")[:500])
    return data.get("response", "")


__all__ = ["ollama_generate", "LLMError"]
