from __future__ import annotations

from typing import Any, Optional

from homie_core.storage.database import Database
from homie_core.utils import utc_now


class FeedbackCollector:
    CHANNELS = ("correction", "preference", "teaching", "satisfaction", "onboarding")

    def __init__(self, db: Database):
        self._db = db

    def record_correction(self, wrong: str, right: str, context: dict | None = None) -> int:
        return self._db.record_feedback(
            channel="correction",
            content=f"Wrong: {wrong} | Right: {right}",
            context={**(context or {}), "wrong": wrong, "right": right},
        )

    def record_preference(self, action: str, accepted: bool, context: dict | None = None) -> int:
        return self._db.record_feedback(
            channel="preference",
            content=f"{'Accepted' if accepted else 'Dismissed'}: {action}",
            context={**(context or {}), "action": action, "accepted": accepted},
        )

    def record_teaching(self, fact: str, context: dict | None = None) -> int:
        return self._db.record_feedback(
            channel="teaching",
            content=fact,
            context=context,
        )

    def record_satisfaction(self, action: str, rating: str, context: dict | None = None) -> int:
        return self._db.record_feedback(
            channel="satisfaction",
            content=f"{rating}: {action}",
            context={**(context or {}), "action": action, "rating": rating},
        )

    def record_onboarding(self, question: str, answer: str) -> int:
        return self._db.record_feedback(
            channel="onboarding",
            content=f"Q: {question} | A: {answer}",
            context={"question": question, "answer": answer},
        )

    def get_recent(self, channel: str | None = None, limit: int = 50) -> list[dict]:
        return self._db.get_recent_feedback(limit=limit, channel=channel)

    def get_correction_count(self, topic: str | None = None) -> int:
        corrections = self._db.get_recent_feedback(limit=1000, channel="correction")
        if topic:
            return sum(1 for c in corrections if topic.lower() in c["content"].lower())
        return len(corrections)
