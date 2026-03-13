from __future__ import annotations

from typing import Any

from homie_core.feedback.beliefs import BeliefSystem
from homie_core.feedback.patterns import PatternDetector


class BehavioralAdapter:
    def __init__(self, belief_system: BeliefSystem, pattern_detector: PatternDetector):
        self._beliefs = belief_system
        self._patterns = pattern_detector
        self._suggestion_threshold: float = 0.6

    def should_suggest(self, action: str, context: dict | None = None) -> bool:
        pref_patterns = self._patterns.find_preference_patterns()
        if action in pref_patterns:
            acceptance_rate = pref_patterns[action]
            if acceptance_rate < 0.3:
                return False
        relevant = self._beliefs.find_belief(action)
        if relevant:
            avg_conf = sum(b["confidence"] for b in relevant) / len(relevant)
            return avg_conf >= self._suggestion_threshold
        return True

    def get_suggestion_timing(self) -> str:
        deep_work_beliefs = self._beliefs.find_belief("deep work") + self._beliefs.find_belief("interruption")
        for b in deep_work_beliefs:
            if "dislikes" in b["belief"].lower() or "no interruption" in b["belief"].lower():
                if b["confidence"] > 0.7:
                    return "wait_for_break"
        return "immediate"

    def adjust_threshold(self, direction: str) -> float:
        if direction == "more_suggestions":
            self._suggestion_threshold = max(0.3, self._suggestion_threshold - 0.05)
        elif direction == "fewer_suggestions":
            self._suggestion_threshold = min(0.9, self._suggestion_threshold + 0.05)
        return self._suggestion_threshold
