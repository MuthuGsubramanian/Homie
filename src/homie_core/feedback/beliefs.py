from __future__ import annotations

from typing import Any, Optional

from homie_core.storage.database import Database
from homie_core.utils import utc_now


class BeliefSystem:
    def __init__(self, db: Database):
        self._db = db

    def add_belief(self, belief: str, confidence: float, context_tags: list[str] | None = None, source_count: int = 1) -> int:
        return self._db.store_belief(belief, confidence, source_count=source_count, context_tags=context_tags)

    def get_beliefs(self, min_confidence: float = 0.0) -> list[dict]:
        return self._db.get_beliefs(min_confidence=min_confidence)

    def update_confidence(self, belief_id: int, new_confidence: float) -> None:
        self._db._conn.execute(
            "UPDATE beliefs SET confidence = ?, last_updated = ? WHERE id = ?",
            (min(1.0, max(0.0, new_confidence)), utc_now().isoformat(), belief_id),
        )
        self._db._conn.commit()

    def reinforce(self, belief_id: int, boost: float = 0.05) -> None:
        beliefs = self.get_beliefs()
        for b in beliefs:
            if b["id"] == belief_id:
                new_conf = min(1.0, b["confidence"] + boost)
                self.update_confidence(belief_id, new_conf)
                self._db._conn.execute(
                    "UPDATE beliefs SET source_count = source_count + 1 WHERE id = ?", (belief_id,)
                )
                self._db._conn.commit()
                return

    def weaken(self, belief_id: int, penalty: float = 0.1) -> None:
        beliefs = self.get_beliefs()
        for b in beliefs:
            if b["id"] == belief_id:
                new_conf = max(0.0, b["confidence"] - penalty)
                self.update_confidence(belief_id, new_conf)
                return

    def decay_all(self, rate: float = 0.01) -> None:
        beliefs = self.get_beliefs()
        for b in beliefs:
            new_conf = max(0.0, b["confidence"] - rate)
            self._db._conn.execute(
                "UPDATE beliefs SET confidence = ? WHERE id = ?", (new_conf, b["id"])
            )
        self._db._conn.commit()

    def find_belief(self, keyword: str) -> list[dict]:
        beliefs = self.get_beliefs()
        return [b for b in beliefs if keyword.lower() in b["belief"].lower()]
