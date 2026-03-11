"""Thompson Sampling suggestion ranker using Beta-Bernoulli bandits.

Each suggestion type (break, workflow, anomaly, etc.) is modeled as a
Bernoulli bandit with a Beta(alpha, beta) prior. Thompson sampling
draws from each posterior to combine learned acceptance rates with
raw confidence scores for ranking.

This balances exploration (trying underrepresented types) with
exploitation (favoring types the user accepts).

Reference: Thompson (1933), Chapelle & Li (2011)
"""
from __future__ import annotations
import random
from collections import defaultdict
from typing import Optional


class SuggestionRanker:
    def __init__(self, seed: int = 42, decay_rate: float = 0.01):
        self._rng = random.Random(seed)
        self._decay_rate = decay_rate
        # Beta distribution params per suggestion type
        # Prior: Beta(1, 1) = uniform
        self._stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"alpha": 1.0, "beta": 1.0}
        )

    def _thompson_score(self, stype: str, stats: Optional[dict] = None) -> float:
        """Draw from Beta(alpha, beta) posterior for this type."""
        s = (stats or self._stats).get(stype, {"alpha": 1.0, "beta": 1.0})
        # Use random.betavariate for Thompson sampling
        return self._rng.betavariate(s["alpha"], s["beta"])

    def rank(
        self, items: list[dict], top_n: Optional[int] = None,
        type_stats: Optional[dict] = None,
    ) -> list[dict]:
        """Rank suggestion items by Thompson-sampled score * confidence.

        Each item must have 'type' and 'confidence' keys.
        Returns items sorted by combined score.
        """
        if not items:
            return []

        scored = []
        for item in items:
            thompson = self._thompson_score(item["type"], type_stats)
            combined = 0.5 * thompson + 0.5 * item.get("confidence", 0.5)
            scored.append((combined, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [item for _, item in scored]
        if top_n is not None:
            return result[:top_n]
        return result

    def record_outcome(self, stype: str, accepted: bool) -> None:
        """Update Beta prior based on user outcome."""
        if stype not in self._stats:
            self._stats[stype] = {"alpha": 1.0, "beta": 1.0}
        if accepted:
            self._stats[stype]["alpha"] += 1.0
        else:
            self._stats[stype]["beta"] += 1.0

    def apply_decay(self) -> None:
        """Apply decay to move priors back toward uniform.

        Prevents old data from dominating forever.
        """
        for stype in self._stats:
            s = self._stats[stype]
            s["alpha"] = max(1.0, s["alpha"] * (1.0 - self._decay_rate))
            s["beta"] = max(1.0, s["beta"] * (1.0 - self._decay_rate))

    def get_type_stats(self, stype: str) -> dict[str, float]:
        """Get current Beta parameters for a type."""
        return dict(self._stats.get(stype, {"alpha": 1.0, "beta": 1.0}))

    def serialize(self) -> dict:
        return {
            "decay_rate": self._decay_rate,
            "stats": {k: dict(v) for k, v in self._stats.items()},
        }

    @classmethod
    def deserialize(cls, data: dict) -> SuggestionRanker:
        ranker = cls(decay_rate=data.get("decay_rate", 0.01))
        for k, v in data.get("stats", {}).items():
            ranker._stats[k] = v
        return ranker
