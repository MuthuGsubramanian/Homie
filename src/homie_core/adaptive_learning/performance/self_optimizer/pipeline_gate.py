"""Pipeline gate — overrides complexity tier to skip unnecessary cognitive stages."""

import threading
from collections import defaultdict
from typing import Optional

# Tier promotion order
_TIER_ORDER = ["trivial", "simple", "moderate", "complex", "deep"]


class PipelineGate:
    """Overrides query complexity tier based on learned quality feedback."""

    def __init__(self, promotion_threshold: int = 3) -> None:
        self._threshold = promotion_threshold
        self._lock = threading.Lock()
        # {original_tier: promoted_tier}
        self._promotions: dict[str, str] = {}
        # {tier: clarification_count}
        self._clarification_counts: dict[str, int] = defaultdict(int)
        # Stats
        self._apply_counts: dict[str, int] = defaultdict(int)

    def apply(self, complexity: str) -> str:
        """Apply gating — returns the effective complexity tier."""
        with self._lock:
            self._apply_counts[complexity] += 1
            # Check if this tier has been promoted
            effective = self._promotions.get(complexity, complexity)
            return effective

    def record_clarification(self, tier: str) -> None:
        """Record that a gated query at this tier needed clarification."""
        with self._lock:
            self._clarification_counts[tier] += 1
            if self._clarification_counts[tier] >= self._threshold:
                # Promote to next tier
                idx = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else -1
                if idx >= 0 and idx < len(_TIER_ORDER) - 1:
                    promoted = _TIER_ORDER[idx + 1]
                    self._promotions[tier] = promoted
                    self._clarification_counts[tier] = 0  # reset

    def get_stats(self) -> dict[str, dict]:
        """Get gating statistics."""
        with self._lock:
            return {
                tier: {
                    "apply_count": self._apply_counts.get(tier, 0),
                    "clarifications": self._clarification_counts.get(tier, 0),
                    "promoted_to": self._promotions.get(tier),
                }
                for tier in _TIER_ORDER
                if self._apply_counts.get(tier, 0) > 0 or tier in self._promotions
            }
