from __future__ import annotations

from collections import defaultdict
from typing import Any

from homie_core.storage.database import Database


class PatternDetector:
    def __init__(self, db: Database):
        self._db = db

    def find_correction_clusters(self, min_count: int = 3) -> list[dict[str, Any]]:
        corrections = self._db.get_recent_feedback(limit=500, channel="correction")
        clusters: dict[str, list] = defaultdict(list)
        for c in corrections:
            ctx = c.get("context", {})
            key = ctx.get("wrong", c["content"][:50])
            clusters[key].append(c)
        return [
            {"topic": k, "count": len(v), "examples": v[:3]}
            for k, v in clusters.items()
            if len(v) >= min_count
        ]

    def find_preference_patterns(self) -> dict[str, float]:
        prefs = self._db.get_recent_feedback(limit=500, channel="preference")
        action_stats: dict[str, dict] = defaultdict(lambda: {"accepted": 0, "total": 0})
        for p in prefs:
            ctx = p.get("context", {})
            action = ctx.get("action", "unknown")
            action_stats[action]["total"] += 1
            if ctx.get("accepted"):
                action_stats[action]["accepted"] += 1
        return {
            action: stats["accepted"] / stats["total"]
            for action, stats in action_stats.items()
            if stats["total"] >= 2
        }

    def find_temporal_patterns(self) -> list[dict]:
        feedback = self._db.get_recent_feedback(limit=500)
        hourly: dict[int, int] = defaultdict(int)
        for f in feedback:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(f["created_at"])
                hourly[dt.hour] += 1
            except (ValueError, KeyError):
                pass
        if not hourly:
            return []
        peak_hour = max(hourly, key=hourly.get)
        return [{"type": "peak_feedback_hour", "hour": peak_hour, "count": hourly[peak_hour]}]
