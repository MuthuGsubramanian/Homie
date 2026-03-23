# src/homie_core/meta_learning/performance_tracker.py
"""Meta Performance Tracker — monitors how well Homie is improving over time."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_DAY_SECONDS = 86_400


@dataclass
class _TaskEntry:
    """Single task completion record."""

    task_type: str
    timestamp: float
    duration_ms: float
    success: bool
    quality_score: float


class MetaPerformanceTracker:
    """Tracks how well Homie is improving over time."""

    def __init__(self, storage: Any | None = None):
        self._storage = storage
        self._entries: list[_TaskEntry] = []

    # ── recording ───────────────────────────────────────────────────────

    def record_task(
        self,
        task_type: str,
        duration_ms: float,
        success: bool,
        quality_score: float,
    ) -> None:
        """Record a task completion for trend analysis."""
        self._entries.append(
            _TaskEntry(
                task_type=task_type,
                timestamp=time.time(),
                duration_ms=duration_ms,
                success=success,
                quality_score=max(0.0, min(1.0, quality_score)),
            )
        )

    # ── analysis ────────────────────────────────────────────────────────

    def get_improvement_trend(
        self, task_type: str, window_days: int = 30
    ) -> dict:
        """Return trend direction, improvement rate, and confidence for *task_type*.

        Splits the window into two halves and compares success rate + quality.
        """
        cutoff = time.time() - window_days * _DAY_SECONDS
        relevant = [e for e in self._entries if e.task_type == task_type and e.timestamp >= cutoff]

        if len(relevant) < 2:
            return {
                "task_type": task_type,
                "direction": "insufficient_data",
                "improvement_rate": 0.0,
                "confidence": 0.0,
                "sample_size": len(relevant),
            }

        relevant.sort(key=lambda e: e.timestamp)
        mid = len(relevant) // 2
        first_half = relevant[:mid]
        second_half = relevant[mid:]

        sr1 = _success_rate(first_half)
        sr2 = _success_rate(second_half)
        q1 = _avg_quality(first_half)
        q2 = _avg_quality(second_half)

        # Composite change (weighted 60 % success-rate, 40 % quality)
        delta = 0.6 * (sr2 - sr1) + 0.4 * (q2 - q1)

        if delta > 0.02:
            direction = "improving"
        elif delta < -0.02:
            direction = "declining"
        else:
            direction = "stable"

        confidence = min(1.0, len(relevant) / 50)

        return {
            "task_type": task_type,
            "direction": direction,
            "improvement_rate": round(delta, 4),
            "confidence": round(confidence, 4),
            "sample_size": len(relevant),
        }

    def get_bottlenecks(self) -> list[dict]:
        """Return task types Homie is struggling with, sorted worst-first."""
        by_type: dict[str, list[_TaskEntry]] = {}
        for e in self._entries:
            by_type.setdefault(e.task_type, []).append(e)

        bottlenecks: list[dict] = []
        for task_type, entries in by_type.items():
            sr = _success_rate(entries)
            aq = _avg_quality(entries)
            score = 0.6 * sr + 0.4 * aq
            if score < 0.7:
                bottlenecks.append(
                    {
                        "task_type": task_type,
                        "success_rate": round(sr, 4),
                        "avg_quality": round(aq, 4),
                        "composite_score": round(score, 4),
                        "sample_size": len(entries),
                    }
                )

        bottlenecks.sort(key=lambda b: b["composite_score"])
        return bottlenecks

    def get_overall_health(self) -> dict:
        """Overall system health and improvement metrics."""
        if not self._entries:
            return {
                "status": "no_data",
                "total_tasks": 0,
                "overall_success_rate": 0.0,
                "overall_avg_quality": 0.0,
                "bottleneck_count": 0,
            }

        sr = _success_rate(self._entries)
        aq = _avg_quality(self._entries)
        bottlenecks = self.get_bottlenecks()

        if sr >= 0.9 and aq >= 0.8:
            status = "healthy"
        elif sr >= 0.7:
            status = "fair"
        else:
            status = "needs_attention"

        return {
            "status": status,
            "total_tasks": len(self._entries),
            "overall_success_rate": round(sr, 4),
            "overall_avg_quality": round(aq, 4),
            "bottleneck_count": len(bottlenecks),
        }


# ── helpers ─────────────────────────────────────────────────────────────

def _success_rate(entries: list[_TaskEntry]) -> float:
    if not entries:
        return 0.0
    return sum(1 for e in entries if e.success) / len(entries)


def _avg_quality(entries: list[_TaskEntry]) -> float:
    if not entries:
        return 0.0
    return sum(e.quality_score for e in entries) / len(entries)
