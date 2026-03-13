from __future__ import annotations

import threading
from typing import Any, Optional

from homie_core.memory.episodic import EpisodicMemory
from homie_core.memory.semantic import SemanticMemory


class ProactiveRetrieval:
    """Silently pre-fetches relevant context on every context change."""

    def __init__(self, semantic_memory: Optional[SemanticMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None):
        self._sm = semantic_memory
        self._em = episodic_memory
        self._staged: dict[str, list] = {"facts": [], "episodes": []}
        self._last_key: str = ""
        self._lock = threading.Lock()

    def on_context_change(self, process: str, title: str) -> None:
        key = f"{process}::{title}"
        if key == self._last_key:
            return
        self._last_key = key

        query = self._build_query(process, title)
        facts: list[dict] = []
        episodes: list[dict] = []

        if self._sm:
            try:
                facts = self._sm.get_facts(min_confidence=0.5)
            except Exception:
                facts = []
        if self._em:
            try:
                episodes = self._em.recall(query, n=3)
            except Exception:
                episodes = []

        with self._lock:
            self._staged = {"facts": facts, "episodes": episodes}

    def get_staged_context(self) -> dict[str, list]:
        with self._lock:
            return dict(self._staged)

    def consume_staged_context(self) -> dict[str, list]:
        with self._lock:
            result = dict(self._staged)
            self._staged = {"facts": [], "episodes": []}
            return result

    def _build_query(self, process: str, title: str) -> str:
        parts = title.split(" - ")
        if len(parts) >= 2:
            return f"{parts[0].strip()} {parts[-1].strip()}"
        return title
