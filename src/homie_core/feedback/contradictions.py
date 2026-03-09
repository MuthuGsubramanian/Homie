from __future__ import annotations

from typing import Any

from homie_core.feedback.beliefs import BeliefSystem


class ContradictionResolver:
    def __init__(self, belief_system: BeliefSystem):
        self._beliefs = belief_system

    def detect_contradictions(self, new_belief: str, new_context: list[str] | None = None) -> list[dict]:
        existing = self._beliefs.get_beliefs()
        contradictions = []
        new_lower = new_belief.lower()
        negation_pairs = [
            ("likes", "dislikes"), ("prefers", "avoids"), ("wants", "doesn't want"),
            ("enjoys", "hates"), ("always", "never"), ("enables", "disables"),
        ]
        for b in existing:
            existing_lower = b["belief"].lower()
            is_contradiction = False
            for pos, neg in negation_pairs:
                if (pos in new_lower and neg in existing_lower) or (neg in new_lower and pos in existing_lower):
                    base_new = new_lower.replace(pos, "").replace(neg, "")
                    base_existing = existing_lower.replace(pos, "").replace(neg, "")
                    if self._similarity(base_new, base_existing) > 0.5:
                        is_contradiction = True
                        break
            if is_contradiction:
                contradictions.append({
                    "existing_belief": b,
                    "new_belief": new_belief,
                    "resolution_needed": True,
                })
        return contradictions

    def resolve(self, existing_belief_id: int, new_belief: str, strategy: str = "replace", new_context: list[str] | None = None) -> dict:
        if strategy == "replace":
            self._beliefs.weaken(existing_belief_id, penalty=1.0)
            new_id = self._beliefs.add_belief(new_belief, confidence=0.7, context_tags=new_context)
            return {"action": "replaced", "new_id": new_id}
        elif strategy == "context_split":
            self._beliefs.add_belief(new_belief, confidence=0.7, context_tags=new_context)
            return {"action": "context_split", "note": "Both beliefs kept with different contexts"}
        elif strategy == "temporal":
            self._beliefs.weaken(existing_belief_id, penalty=0.3)
            new_id = self._beliefs.add_belief(new_belief, confidence=0.7, context_tags=new_context)
            return {"action": "temporal_evolution", "new_id": new_id}
        return {"action": "none"}

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
