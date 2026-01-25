from dataclasses import dataclass
from typing import Dict, List, Any
import yaml
from pathlib import Path

@dataclass
class HomieConfig:
    raw: Dict[str, Any]

def load_config(path: str = "homie.config.yaml") -> HomieConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config: {p.resolve()}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return HomieConfig(raw=data)

def cfg_get(cfg: HomieConfig, *keys, default=None):
    cur = cfg.raw
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
