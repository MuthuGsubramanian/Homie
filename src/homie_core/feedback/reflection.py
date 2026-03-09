from __future__ import annotations

import json
from typing import Any

from homie_core.feedback.beliefs import BeliefSystem
from homie_core.feedback.patterns import PatternDetector
from homie_core.storage.database import Database
from homie_core.utils import utc_now


class ReflectionEngine:
    def __init__(self, db: Database, belief_system: BeliefSystem, pattern_detector: PatternDetector, model_engine=None):
        self._db = db
        self._beliefs = belief_system
        self._patterns = pattern_detector
        self._engine = model_engine

    def generate_reflection(self) -> dict[str, Any]:
        beliefs = self._beliefs.get_beliefs()
        pref_patterns = self._patterns.find_preference_patterns()
        correction_clusters = self._patterns.find_correction_clusters(min_count=2)

        reflection = {
            "timestamp": utc_now().isoformat(),
            "total_beliefs": len(beliefs),
            "high_confidence_beliefs": len([b for b in beliefs if b["confidence"] > 0.8]),
            "low_confidence_beliefs": len([b for b in beliefs if b["confidence"] < 0.3]),
            "preference_patterns": pref_patterns,
            "correction_clusters": len(correction_clusters),
            "insights": [],
        }

        # Generate insights
        if correction_clusters:
            reflection["insights"].append(
                f"User has corrected me {len(correction_clusters)} times on recurring topics — need to update behavior."
            )

        low_acceptance = {k: v for k, v in pref_patterns.items() if v < 0.3}
        if low_acceptance:
            actions = ", ".join(low_acceptance.keys())
            reflection["insights"].append(
                f"Low acceptance rate for: {actions}. Consider reducing these suggestions."
            )

        stale = [b for b in beliefs if b["confidence"] < 0.2]
        if stale:
            reflection["insights"].append(
                f"{len(stale)} beliefs have very low confidence — consider removing or re-validating."
            )

        return reflection

    def get_acceptance_trend(self, days: int = 7) -> float:
        prefs = self._db.get_recent_feedback(limit=200, channel="preference")
        if not prefs:
            return 0.0
        accepted = sum(1 for p in prefs if p.get("context", {}).get("accepted", False))
        return accepted / len(prefs)
