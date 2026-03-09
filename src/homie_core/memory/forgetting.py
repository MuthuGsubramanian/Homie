from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from homie_core.storage.database import Database
from homie_core.utils import utc_now


class ForgettingCurve:
    def __init__(self, db: Database, decay_rate: float = 0.1):
        self._db = db
        self._decay_rate = decay_rate

    def calculate_relevance(self, base_score: float, last_accessed: str, access_count: int) -> float:
        try:
            last_dt = datetime.fromisoformat(last_accessed)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return base_score
        now = utc_now()
        days_since = (now - last_dt).total_seconds() / 86400.0
        recency_factor = math.exp(-self._decay_rate * days_since)
        frequency_factor = min(1.0, math.log(access_count + 1) / math.log(20))
        return base_score * recency_factor * (0.5 + 0.5 * frequency_factor)

    def decay_all(self, threshold: float = 0.05) -> int:
        facts = self._db.get_facts(include_archived=False)
        archived_count = 0
        for fact in facts:
            relevance = self.calculate_relevance(
                fact["confidence"], fact["last_confirmed"], fact["source_count"]
            )
            if relevance < threshold:
                self._db._conn.execute(
                    "UPDATE semantic_memory SET archived = 1 WHERE id = ?", (fact["id"],)
                )
                archived_count += 1
        if archived_count:
            self._db._conn.commit()
        return archived_count
