from __future__ import annotations

import json
from typing import Any, Optional

from homie_core.storage.database import Database
from homie_core.utils import utc_now


class SemanticMemory:
    def __init__(self, db: Database):
        self._db = db

    def learn(self, fact: str, confidence: float = 0.5, tags: Optional[list[str]] = None) -> int:
        return self._db.store_fact(fact, confidence=confidence, tags=tags)

    def get_facts(self, min_confidence: float = 0.0) -> list[dict]:
        return self._db.get_facts(min_confidence=min_confidence)

    def reinforce(self, fact: str, boost: float = 0.05) -> None:
        facts = self._db.get_facts()
        for f in facts:
            if f["fact"] == fact:
                new_conf = min(1.0, f["confidence"] + boost)
                self._db._conn.execute(
                    "UPDATE semantic_memory SET confidence = ?, last_confirmed = ?, source_count = source_count + 1 WHERE id = ?",
                    (new_conf, utc_now().isoformat(), f["id"]),
                )
                self._db._conn.commit()
                return

    def forget_topic(self, tag: str) -> None:
        self._db._conn.execute("UPDATE semantic_memory SET archived = 1 WHERE tags LIKE ?", (f'%"{tag}"%',))
        self._db._conn.commit()

    def forget_fact(self, fact_id: int) -> None:
        self._db._conn.execute("UPDATE semantic_memory SET archived = 1 WHERE id = ?", (fact_id,))
        self._db._conn.commit()

    def set_profile(self, domain: str, data: dict[str, Any]) -> None:
        self._db.store_profile(domain, data)

    def get_profile(self, domain: str) -> Optional[dict[str, Any]]:
        return self._db.get_profile(domain)

    def get_all_profiles(self) -> dict[str, dict]:
        rows = self._db._conn.execute("SELECT domain, data FROM profile").fetchall()
        return {r["domain"]: json.loads(r["data"]) for r in rows}
