"""Semantic Drift Detection — tracks how user interests shift over time.

Compares recent conversation topics against historical patterns to detect
when the user's focus areas are changing significantly. Drift detection
helps the consolidation scheduler prioritize which memories to reinforce
and which to let decay.

Operates on topic distributions extracted from episodic memory summaries.
No model calls required — uses lightweight word-frequency analysis.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from homie_core.memory.episodic import EpisodicMemory
from homie_core.utils import utc_now


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Result of a semantic drift analysis."""

    drift_score: float  # 0.0 = no drift, 1.0 = complete topic shift
    emerging_topics: list[str]  # Topics gaining frequency
    fading_topics: list[str]  # Topics losing frequency
    stable_topics: list[str]  # Topics with consistent presence
    timestamp: str = ""
    window_recent: int = 0  # Number of recent episodes analyzed
    window_historical: int = 0  # Number of historical episodes analyzed

    @property
    def is_significant(self) -> bool:
        """Drift score above 0.4 indicates a meaningful shift."""
        return self.drift_score > 0.4


# ---------------------------------------------------------------------------
# Stop words for topic extraction (shared with learning_pipeline style)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "so", "if", "then", "than", "too", "very", "just",
    "about", "up", "out", "that", "this", "it", "its", "my", "me", "i",
    "you", "your", "we", "our", "they", "their", "what", "which", "who",
    "how", "when", "where", "why", "all", "each", "every", "both",
    "some", "any", "other", "more", "most", "such", "here", "there",
    "user", "session", "turn", "topics", "learned", "general",
})


def _extract_topic_distribution(summaries: list[str]) -> Counter:
    """Build a word-frequency distribution from episode summaries."""
    counts: Counter = Counter()
    for summary in summaries:
        words = re.findall(r"\w+", summary.lower())
        for word in words:
            if word not in _STOP_WORDS and len(word) > 2:
                counts[word] += 1
    return counts


def _jensen_shannon_divergence(p: Counter, q: Counter) -> float:
    """Compute Jensen-Shannon divergence between two distributions.

    Returns a value in [0, 1]. 0 = identical distributions,
    1 = completely different.
    """
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0

    total_p = sum(p.values()) or 1
    total_q = sum(q.values()) or 1

    jsd = 0.0
    for key in all_keys:
        p_val = p.get(key, 0) / total_p
        q_val = q.get(key, 0) / total_q
        m_val = (p_val + q_val) / 2

        if p_val > 0 and m_val > 0:
            jsd += 0.5 * p_val * math.log2(p_val / m_val)
        if q_val > 0 and m_val > 0:
            jsd += 0.5 * q_val * math.log2(q_val / m_val)

    # Clamp to [0, 1] for numerical safety
    return max(0.0, min(1.0, jsd))


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------

class SemanticDriftDetector:
    """Tracks how the user's interests and topics shift over time.

    Compares a recent window of episodes against a historical window
    using Jensen-Shannon divergence on topic word distributions.
    Lightweight — no model calls, runs in < 100ms typically.
    """

    def __init__(
        self,
        episodic_memory: Optional[EpisodicMemory] = None,
        recent_window: int = 10,
        historical_window: int = 50,
        significance_threshold: float = 0.4,
    ):
        self._em = episodic_memory
        self._recent_window = recent_window
        self._historical_window = historical_window
        self._significance_threshold = significance_threshold
        self._history: list[DriftReport] = []

    def analyze(self) -> DriftReport:
        """Run drift analysis comparing recent vs historical topics.

        Returns a DriftReport with the drift score and topic changes.
        """
        if not self._em:
            report = DriftReport(
                drift_score=0.0,
                emerging_topics=[],
                fading_topics=[],
                stable_topics=[],
                timestamp=utc_now().isoformat(),
            )
            self._history.append(report)
            if len(self._history) > 50:
                self._history = self._history[-50:]
            return report

        # Fetch episodes — recent and historical windows
        try:
            all_episodes = self._em.recall("", n=self._historical_window)
        except Exception:
            report = DriftReport(
                drift_score=0.0,
                emerging_topics=[],
                fading_topics=[],
                stable_topics=[],
                timestamp=utc_now().isoformat(),
            )
            self._history.append(report)
            if len(self._history) > 50:
                self._history = self._history[-50:]
            return report

        if len(all_episodes) < self._recent_window + 5:
            # Not enough data for meaningful drift detection
            return DriftReport(
                drift_score=0.0,
                emerging_topics=[],
                fading_topics=[],
                stable_topics=[],
                timestamp=utc_now().isoformat(),
                window_recent=min(len(all_episodes), self._recent_window),
                window_historical=len(all_episodes),
            )

        # Split into recent and historical
        recent_summaries = [
            ep.get("summary", "") for ep in all_episodes[: self._recent_window]
        ]
        historical_summaries = [
            ep.get("summary", "") for ep in all_episodes[self._recent_window :]
        ]

        # Build topic distributions
        recent_dist = _extract_topic_distribution(recent_summaries)
        historical_dist = _extract_topic_distribution(historical_summaries)

        # Compute drift score
        drift_score = _jensen_shannon_divergence(recent_dist, historical_dist)

        # Identify emerging, fading, and stable topics
        recent_top = set(w for w, _ in recent_dist.most_common(20))
        historical_top = set(w for w, _ in historical_dist.most_common(20))

        emerging = sorted(recent_top - historical_top)
        fading = sorted(historical_top - recent_top)
        stable = sorted(recent_top & historical_top)

        report = DriftReport(
            drift_score=drift_score,
            emerging_topics=emerging[:10],
            fading_topics=fading[:10],
            stable_topics=stable[:10],
            timestamp=utc_now().isoformat(),
            window_recent=len(recent_summaries),
            window_historical=len(historical_summaries),
        )

        self._history.append(report)
        # Keep only last 50 reports
        if len(self._history) > 50:
            self._history = self._history[-50:]

        return report

    def get_drift_trend(self, last_n: int = 10) -> list[float]:
        """Return recent drift scores to show trend over time."""
        return [r.drift_score for r in self._history[-last_n:]]

    @property
    def latest_report(self) -> Optional[DriftReport]:
        """Most recent drift report, or None if never analyzed."""
        return self._history[-1] if self._history else None
