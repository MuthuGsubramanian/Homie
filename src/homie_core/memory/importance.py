"""Memory Importance Scoring — not all memories are equal.

Scores memories on a composite scale considering:
- Recency: How recently the memory was accessed or created
- Frequency: How often the memory has been accessed or reinforced
- Emotional significance: Detected from language in the memory content
- Connection count: How many other memories reference similar topics

The composite score drives consolidation decisions: which memories to
keep, merge, or prune.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from homie_core.utils import utc_now


# ---------------------------------------------------------------------------
# Emotional signal detection
# ---------------------------------------------------------------------------

_EMOTION_WORDS: dict[str, float] = {
    # High emotional significance (positive)
    "love": 0.9, "loves": 0.9, "loved": 0.85,
    "amazing": 0.8, "incredible": 0.8, "wonderful": 0.8,
    "excited": 0.8, "thrilled": 0.85, "passionate": 0.85, "promoted": 0.9,
    "breakthrough": 0.85, "milestone": 0.8, "achieved": 0.75,
    # High emotional significance (negative)
    "hate": 0.9, "terrible": 0.8, "disaster": 0.85, "furious": 0.85,
    "devastated": 0.9, "fired": 0.9, "lost": 0.7, "failed": 0.7,
    "frustrated": 0.7, "overwhelmed": 0.75, "stressed": 0.7,
    "crisis": 0.85, "emergency": 0.9, "urgent": 0.75,
    # Medium emotional significance
    "happy": 0.6, "great": 0.5, "good": 0.3, "nice": 0.3,
    "helpful": 0.4, "useful": 0.4, "important": 0.6, "critical": 0.7,
    "worried": 0.6, "concerned": 0.55, "annoyed": 0.5, "confused": 0.4,
    "sad": 0.6, "disappointed": 0.6, "surprised": 0.5,
    # Life events (high significance regardless of sentiment)
    "birthday": 0.8, "wedding": 0.9, "baby": 0.9, "moved": 0.7,
    "graduated": 0.85, "hired": 0.8, "interview": 0.7, "deadline": 0.7,
}


def _detect_emotional_significance(text: str) -> float:
    """Score emotional significance of text content.

    Returns float in [0.0, 1.0]. Higher = more emotionally significant.
    """
    if not text:
        return 0.0

    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0

    scores = []
    for word in words:
        if word in _EMOTION_WORDS:
            scores.append(_EMOTION_WORDS[word])

    if not scores:
        return 0.0

    # Use the max emotional word found, with a small boost for multiple signals
    max_score = max(scores)
    multi_signal_boost = min(0.1, len(scores) * 0.02)
    return min(1.0, max_score + multi_signal_boost)


# ---------------------------------------------------------------------------
# Importance scorer
# ---------------------------------------------------------------------------

@dataclass
class ImportanceScore:
    """Breakdown of a memory's importance score."""

    total: float  # Composite score in [0.0, 1.0]
    recency: float
    frequency: float
    emotional: float
    connections: float
    memory_id: str = ""


class MemoryImportanceScorer:
    """Scores memories by composite importance for consolidation decisions.

    Weights can be tuned but default to a balanced distribution that
    slightly favors recency and emotional significance.
    """

    def __init__(
        self,
        recency_weight: float = 0.30,
        frequency_weight: float = 0.20,
        emotional_weight: float = 0.30,
        connection_weight: float = 0.20,
        decay_rate: float = 0.1,
    ):
        self._weights = {
            "recency": recency_weight,
            "frequency": frequency_weight,
            "emotional": emotional_weight,
            "connections": connection_weight,
        }
        self._decay_rate = decay_rate

    def score_memory(
        self,
        memory: dict[str, Any],
        all_memories: Optional[list[dict[str, Any]]] = None,
    ) -> ImportanceScore:
        """Score a single memory's importance.

        Args:
            memory: Dict with keys like 'fact'/'summary', 'confidence',
                    'last_confirmed'/'created_at', 'source_count', 'tags'.
            all_memories: All memories for computing connection count.
                          If None, connection score defaults to 0.

        Returns:
            ImportanceScore with component breakdown.
        """
        recency = self._score_recency(memory)
        frequency = self._score_frequency(memory)
        emotional = self._score_emotional(memory)
        connections = self._score_connections(memory, all_memories or [])

        total = (
            self._weights["recency"] * recency
            + self._weights["frequency"] * frequency
            + self._weights["emotional"] * emotional
            + self._weights["connections"] * connections
        )

        return ImportanceScore(
            total=min(1.0, total),
            recency=recency,
            frequency=frequency,
            emotional=emotional,
            connections=connections,
            memory_id=str(memory.get("id", "")),
        )

    def score_batch(
        self, memories: list[dict[str, Any]]
    ) -> list[ImportanceScore]:
        """Score a batch of memories, computing connections across them."""
        return [self.score_memory(m, memories) for m in memories]

    def _score_recency(self, memory: dict[str, Any]) -> float:
        """Score based on how recently the memory was accessed/confirmed."""
        timestamp_str = memory.get("last_confirmed") or memory.get("created_at", "")
        if not timestamp_str:
            return 0.0

        try:
            last_dt = datetime.fromisoformat(timestamp_str)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return 0.0

        days_since = (utc_now() - last_dt).total_seconds() / 86400.0
        return math.exp(-self._decay_rate * days_since)

    def _score_frequency(self, memory: dict[str, Any]) -> float:
        """Score based on access/reinforcement count."""
        count = memory.get("source_count", 1)
        # Logarithmic scaling — first few accesses matter most
        return min(1.0, math.log(count + 1) / math.log(20))

    def _score_emotional(self, memory: dict[str, Any]) -> float:
        """Score based on emotional significance of the content."""
        text = memory.get("fact", "") or memory.get("summary", "")
        return _detect_emotional_significance(text)

    def _score_connections(
        self,
        memory: dict[str, Any],
        all_memories: list[dict[str, Any]],
    ) -> float:
        """Score based on how many other memories share topics.

        Uses simple word overlap to count connections.
        """
        if not all_memories:
            return 0.0

        text = (memory.get("fact", "") or memory.get("summary", "")).lower()
        words = set(re.findall(r"\w+", text))
        if len(words) < 3:
            return 0.0

        memory_id = memory.get("id")
        connection_count = 0

        for other in all_memories:
            if other.get("id") == memory_id:
                continue
            other_text = (
                other.get("fact", "") or other.get("summary", "")
            ).lower()
            other_words = set(re.findall(r"\w+", other_text))
            overlap = len(words & other_words)
            if overlap >= 3:
                connection_count += 1

        # Normalize: 10+ connections = max score
        return min(1.0, connection_count / 10.0)
