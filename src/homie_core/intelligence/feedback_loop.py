"""Feedback collection and learning signal generation.

Collects user responses to suggestions (accepted, dismissed, snoozed)
and generates learning signals that update the Thompson sampling ranker,
interruption model, and preference engine.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from typing import Optional


class FeedbackType(Enum):
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    SNOOZED = "snoozed"


@dataclass
class FeedbackEvent:
    suggestion_id: str
    suggestion_type: str
    feedback_type: FeedbackType
    reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class FeedbackLoop:
    """Collects and analyzes user feedback on suggestions."""

    def __init__(self):
        self._events: list[FeedbackEvent] = []
        self._counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"accepted": 0, "dismissed": 0, "snoozed": 0}
        )

    @property
    def total_feedback(self) -> int:
        return len(self._events)

    def record(self, event: FeedbackEvent) -> None:
        self._events.append(event)
        self._counts[event.suggestion_type][event.feedback_type.value] += 1

    def acceptance_rate(self, suggestion_type: str) -> float:
        """Acceptance rate for a suggestion type. Returns 0.5 if no data."""
        counts = self._counts.get(suggestion_type)
        if not counts:
            return 0.5
        accepted = counts["accepted"]
        dismissed = counts["dismissed"]
        total = accepted + dismissed
        if total == 0:
            return 0.5
        return accepted / total

    def get_recent(self, n: int = 10) -> list[FeedbackEvent]:
        """Get most recent feedback events."""
        return list(reversed(self._events[-n:]))

    def get_summary(self) -> dict:
        by_type = {}
        for stype, counts in self._counts.items():
            by_type[stype] = dict(counts)
        return {
            "total": len(self._events),
            "by_type": by_type,
        }

    def get_learning_signals(self) -> dict[str, dict[str, int]]:
        """Get per-type learning signals for model updates."""
        return {k: dict(v) for k, v in self._counts.items()}

    def serialize(self) -> dict:
        return {
            "events": [
                {
                    "suggestion_id": e.suggestion_id,
                    "suggestion_type": e.suggestion_type,
                    "feedback_type": e.feedback_type.value,
                    "reason": e.reason,
                    "timestamp": e.timestamp,
                }
                for e in self._events
            ],
        }

    @classmethod
    def deserialize(cls, data: dict) -> FeedbackLoop:
        fl = cls()
        for e_data in data.get("events", []):
            event = FeedbackEvent(
                suggestion_id=e_data["suggestion_id"],
                suggestion_type=e_data["suggestion_type"],
                feedback_type=FeedbackType(e_data["feedback_type"]),
                reason=e_data.get("reason"),
                timestamp=e_data.get("timestamp", 0),
            )
            fl.record(event)
        return fl
