from __future__ import annotations

import uuid
from typing import Any, Optional

from homie_core.storage.database import Database
from homie_core.storage.vectors import VectorStore


class EpisodicMemory:
    def __init__(self, db: Database, vector_store: VectorStore):
        self._db = db
        self._vs = vector_store

    def record(self, summary: str, mood: Optional[str] = None, outcome: Optional[str] = None, context_tags: Optional[list[str]] = None) -> str:
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        self._db.record_episode_meta(summary=summary, mood=mood, outcome=outcome, context_tags=context_tags)
        metadata = {}
        if mood:
            metadata["mood"] = mood
        if outcome:
            metadata["outcome"] = outcome
        if context_tags:
            metadata["tags"] = ",".join(context_tags)
        self._vs.add_episode(episode_id, summary, metadata)
        return episode_id

    def recall(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        results = self._vs.query_episodes(query, n=n)
        enriched = []
        for r in results:
            entry = {
                "id": r["id"],
                "summary": r["text"],
                "mood": r.get("metadata", {}).get("mood"),
                "outcome": r.get("metadata", {}).get("outcome"),
                "distance": r.get("distance"),
            }
            enriched.append(entry)
        return enriched

    def delete(self, episode_ids: list[str]) -> None:
        self._vs.delete_episodes(episode_ids)
