from __future__ import annotations

import json
from pathlib import Path

from homie_core.utils import utc_now


class AuditLogger:
    """Append-only JSON Lines log of all LLM queries and responses."""

    def __init__(self, log_dir: Path | str, enabled: bool = True):
        self._dir = Path(log_dir)
        self._enabled = enabled
        if enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def log_query(self, prompt: str, response: str, model: str) -> None:
        if not self._enabled:
            return
        now = utc_now()
        filename = f"audit_{now.strftime('%Y-%m-%d')}.jsonl"
        entry = {
            "timestamp": now.isoformat(),
            "prompt": prompt,
            "response": response,
            "model": model,
        }
        with open(self._dir / filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
