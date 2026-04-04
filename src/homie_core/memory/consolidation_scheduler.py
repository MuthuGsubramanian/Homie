"""Consolidation Scheduler — runs memory consolidation during idle periods.

Responsibilities:
- Detect idle periods (no active conversation)
- Merge similar episodic memories into semantic facts
- Prune low-importance memories using the importance scorer
- Track consolidation metrics

Designed to be lightweight (< 5 seconds per run) and non-blocking.
Called from the daemon's main loop or scheduler tick.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from homie_core.memory.consolidator import MemoryConsolidator
from homie_core.memory.drift_detector import SemanticDriftDetector
from homie_core.memory.episodic import EpisodicMemory
from homie_core.memory.forgetting import ForgettingCurve
from homie_core.memory.importance import MemoryImportanceScorer
from homie_core.memory.semantic import SemanticMemory
from homie_core.memory.user_model import UserModelSynthesizer
from homie_core.utils import utc_now

logger = logging.getLogger(__name__)

# Minimum seconds between consolidation runs
_MIN_INTERVAL = 300  # 5 minutes
# Maximum episodic memories to process per run
_BATCH_SIZE = 30
# Importance score below which memories are candidates for pruning
_PRUNE_THRESHOLD = 0.15
# Similarity threshold for merging episodic summaries
_MERGE_WORD_OVERLAP = 0.5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ConsolidationMetrics:
    """Tracks consolidation run statistics."""

    total_runs: int = 0
    total_memories_pruned: int = 0
    total_facts_merged: int = 0
    total_user_model_updates: int = 0
    last_run_timestamp: str = ""
    last_run_duration_ms: float = 0.0
    last_drift_score: float = 0.0
    last_memories_scored: int = 0
    last_memories_pruned: int = 0
    last_facts_merged: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "total_memories_pruned": self.total_memories_pruned,
            "total_facts_merged": self.total_facts_merged,
            "total_user_model_updates": self.total_user_model_updates,
            "last_run_timestamp": self.last_run_timestamp,
            "last_run_duration_ms": self.last_run_duration_ms,
            "last_drift_score": self.last_drift_score,
            "last_memories_scored": self.last_memories_scored,
            "last_memories_pruned": self.last_memories_pruned,
            "last_facts_merged": self.last_facts_merged,
        }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class ConsolidationScheduler:
    """Orchestrates memory consolidation during idle periods.

    Performs three operations on each run:
    1. Prune: Score all memories by importance, archive low-scoring ones
    2. Merge: Find similar episodic memories, merge into semantic facts
    3. Update: Refresh the user model with latest memory state
    """

    def __init__(
        self,
        semantic_memory: Optional[SemanticMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        forgetting_curve: Optional[ForgettingCurve] = None,
        drift_detector: Optional[SemanticDriftDetector] = None,
        importance_scorer: Optional[MemoryImportanceScorer] = None,
        user_model: Optional[UserModelSynthesizer] = None,
        min_interval: float = _MIN_INTERVAL,
    ):
        self._sm = semantic_memory
        self._em = episodic_memory
        self._fc = forgetting_curve
        self._drift = drift_detector
        self._scorer = importance_scorer or MemoryImportanceScorer()
        self._user_model = user_model
        self._min_interval = min_interval

        self._last_run: float = 0.0
        self._last_conversation_time: float = 0.0
        self._metrics = ConsolidationMetrics()

    @property
    def metrics(self) -> ConsolidationMetrics:
        return self._metrics

    def notify_conversation_activity(self) -> None:
        """Call this when the user sends a message — resets idle timer."""
        self._last_conversation_time = time.monotonic()

    def is_idle(self, idle_threshold: float = 120.0) -> bool:
        """Check if we're in an idle period (no conversation activity).

        Args:
            idle_threshold: Seconds of silence before considered idle.
        """
        if self._last_conversation_time == 0.0:
            # No conversation yet — consider idle
            return True
        elapsed = time.monotonic() - self._last_conversation_time
        return elapsed >= idle_threshold

    def should_run(self) -> bool:
        """Check if consolidation should run now."""
        if not self.is_idle():
            return False
        elapsed_since_run = time.monotonic() - self._last_run
        return elapsed_since_run >= self._min_interval

    def run(self, force: bool = False) -> ConsolidationMetrics:
        """Run a consolidation cycle.

        Args:
            force: If True, skip idle and interval checks.

        Returns:
            Updated metrics after the run.
        """
        if not force and not self.should_run():
            return self._metrics

        start = time.monotonic()
        pruned = 0
        merged = 0

        try:
            # 1. Run drift detection
            drift_score = self._run_drift_detection()

            # 2. Prune low-importance semantic memories
            pruned = self._run_pruning()

            # 3. Merge similar episodic memories into facts
            merged = self._run_merging()

            # 4. Update user model
            self._run_user_model_update()

        except Exception:
            logger.exception("Consolidation run failed")

        elapsed_ms = (time.monotonic() - start) * 1000

        # Update metrics
        self._metrics.total_runs += 1
        self._metrics.total_memories_pruned += pruned
        self._metrics.total_facts_merged += merged
        self._metrics.last_run_timestamp = utc_now().isoformat()
        self._metrics.last_run_duration_ms = round(elapsed_ms, 1)
        self._metrics.last_drift_score = drift_score
        self._metrics.last_memories_pruned = pruned
        self._metrics.last_facts_merged = merged

        self._last_run = time.monotonic()

        logger.info(
            "Consolidation run #%d: pruned=%d, merged=%d, drift=%.3f, took=%.1fms",
            self._metrics.total_runs, pruned, merged, drift_score, elapsed_ms,
        )

        return self._metrics

    def _run_drift_detection(self) -> float:
        """Run semantic drift analysis. Returns drift score."""
        if not self._drift:
            return 0.0
        try:
            report = self._drift.analyze()
            return report.drift_score
        except Exception:
            logger.debug("Drift detection failed", exc_info=True)
            return 0.0

    def _run_pruning(self) -> int:
        """Score semantic memories and archive low-importance ones."""
        # Run forgetting curve decay if available (independent of fact scoring)
        fc_pruned = 0
        if self._fc:
            try:
                fc_pruned = self._fc.decay_all(threshold=0.05)
            except Exception:
                pass

        if not self._sm:
            return fc_pruned

        try:
            facts = self._sm.get_facts(min_confidence=0.0)
        except Exception:
            return fc_pruned

        if not facts:
            return fc_pruned

        self._metrics.last_memories_scored = len(facts)

        # Score all memories
        scores = self._scorer.score_batch(facts)

        # Archive memories below importance threshold
        importance_pruned = 0
        for fact, score in zip(facts, scores):
            if score.total < _PRUNE_THRESHOLD and fact.get("confidence", 1.0) < 0.4:
                try:
                    self._sm.forget_fact(fact["id"])
                    importance_pruned += 1
                except Exception:
                    pass

        return fc_pruned + importance_pruned

    def _run_merging(self) -> int:
        """Find similar episodic memories and merge into semantic facts."""
        if not self._em or not self._sm:
            return 0

        try:
            episodes = self._em.recall("", n=_BATCH_SIZE)
        except Exception:
            return 0

        if len(episodes) < 2:
            return 0

        # Group episodes by topic similarity using word overlap
        merged_count = 0
        processed_ids: set[str] = set()

        for i, ep_a in enumerate(episodes):
            ep_a_id = ep_a.get("id", "")
            if ep_a_id in processed_ids:
                continue

            summary_a = ep_a.get("summary", "")
            words_a = set(re.findall(r"\w+", summary_a.lower()))
            if len(words_a) < 3:
                continue

            similar_group = [ep_a]

            for j in range(i + 1, len(episodes)):
                ep_b = episodes[j]
                ep_b_id = ep_b.get("id", "")
                if ep_b_id in processed_ids:
                    continue

                summary_b = ep_b.get("summary", "")
                words_b = set(re.findall(r"\w+", summary_b.lower()))
                if len(words_b) < 3:
                    continue

                overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
                if overlap >= _MERGE_WORD_OVERLAP:
                    similar_group.append(ep_b)
                    processed_ids.add(ep_b_id)

            # Merge groups of 3+ similar episodes into a semantic fact
            if len(similar_group) >= 3:
                merged_fact = self._merge_episodes(similar_group)
                if merged_fact:
                    try:
                        self._sm.learn(
                            merged_fact,
                            confidence=0.6,
                            tags=["consolidated", "auto_merged"],
                        )
                        merged_count += 1

                        # Delete the source episodes (keep the first as reference)
                        delete_ids = [
                            ep.get("id")
                            for ep in similar_group[1:]
                            if ep.get("id")
                        ]
                        if delete_ids:
                            self._em.delete(delete_ids)
                    except Exception:
                        logger.debug("Failed to store merged fact", exc_info=True)

            processed_ids.add(ep_a_id)

        return merged_count

    def _merge_episodes(self, episodes: list[dict]) -> Optional[str]:
        """Create a semantic fact from a group of similar episodes.

        Uses the most common keywords across the episodes to build
        a concise factual statement.
        """
        summaries = [ep.get("summary", "") for ep in episodes if ep.get("summary")]
        if not summaries:
            return None

        # Find common keywords
        word_counts: dict[str, int] = {}
        for summary in summaries:
            words = set(re.findall(r"\w+", summary.lower()))
            for word in words:
                if len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Get words appearing in most summaries
        threshold = len(summaries) * 0.5
        common_words = sorted(
            [w for w, c in word_counts.items() if c >= threshold],
            key=lambda w: word_counts[w],
            reverse=True,
        )[:5]

        if not common_words:
            return None

        return (
            f"User frequently engages with topics: {', '.join(common_words)} "
            f"(observed in {len(episodes)} sessions)"
        )

    def _run_user_model_update(self) -> None:
        """Refresh the user model cache."""
        if not self._user_model:
            return
        try:
            self._user_model.get_profile(force_refresh=True)
            self._metrics.total_user_model_updates += 1
        except Exception:
            logger.debug("User model update failed", exc_info=True)
