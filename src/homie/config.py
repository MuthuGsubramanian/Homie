from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


DEFAULT_CONFIG_NAME = "homie.config.yaml"


@dataclass
class HomieConfig:
    """Container for loaded configuration."""

    raw: Dict[str, Any]
    path: Path

    def get(self, *keys: str, default: Any = None) -> Any:
        return cfg_get(self, *keys, default=default)


def cfg_get(cfg: HomieConfig, *keys: str, default: Any = None) -> Any:
    cur: Any = cfg.raw
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides without mutating the original dict."""
    updated = dict(data)
    llm_cfg = dict(updated.get("llm", {}))

    model_override = os.getenv("HOMIE_MODEL")
    base_url_override = os.getenv("HOMIE_OLLAMA_URL")

    if model_override:
        llm_cfg["model"] = model_override
    if base_url_override:
        llm_cfg["base_url"] = base_url_override

    if llm_cfg:
        updated["llm"] = llm_cfg

    return updated


def _resolve_config_path(path: Optional[str | Path]) -> Path:
    env_path = os.getenv("HOMIE_CONFIG")
    candidate = Path(path or env_path or DEFAULT_CONFIG_NAME)

    if not candidate.is_absolute():
        # Prefer CWD
        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            return cwd_candidate.resolve()
        # Fallback to package root
        pkg_root = Path(__file__).resolve().parent.parent
        pkg_candidate = pkg_root / candidate
        if pkg_candidate.exists():
            return pkg_candidate.resolve()
    return candidate.resolve()


def load_config(path: Optional[str | Path] = None) -> HomieConfig:
    cfg_path = _resolve_config_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing configuration file at {cfg_path}")

    raw_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw_data = _apply_env_overrides(raw_data)
    return HomieConfig(raw=raw_data, path=cfg_path)


def list_targets(cfg: HomieConfig) -> Iterable[str]:
    targets = cfg_get(cfg, "ssh", "targets", default={}) or {}
    return targets.keys()


__all__ = [
    "HomieConfig",
    "cfg_get",
    "load_config",
    "list_targets",
]
