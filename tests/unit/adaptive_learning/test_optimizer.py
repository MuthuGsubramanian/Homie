# tests/unit/adaptive_learning/test_optimizer.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.performance.optimizer import PerformanceOptimizer
from homie_core.adaptive_learning.observation.signals import (
    LearningSignal,
    SignalType,
    SignalCategory,
)


class TestPerformanceOptimizer:
    def _make_optimizer(self, storage=None):
        storage = storage or MagicMock()
        return PerformanceOptimizer(storage=storage, cache_max_entries=100, cache_ttl=3600)

    def test_cache_response(self):
        opt = self._make_optimizer()
        opt.cache_response("What is Python?", "A language.", context_hash="ctx1")
        result = opt.get_cached_response("What is Python?")
        assert result is not None

    def test_cache_miss(self):
        opt = self._make_optimizer()
        assert opt.get_cached_response("unknown") is None

    def test_on_signal_records_activity(self):
        opt = self._make_optimizer()
        sig = LearningSignal(
            signal_type=SignalType.BEHAVIORAL,
            category=SignalCategory.PERFORMANCE,
            source="system",
            data={"hour": 9, "activity": "inference"},
            context={},
        )
        opt.on_signal(sig)
        assert opt.resource_scheduler.predict_activity(9) == "inference"

    def test_rank_context_sources(self):
        opt = self._make_optimizer()
        opt.context_optimizer.record_usage("coding", "git", was_referenced=True)
        opt.context_optimizer.record_usage("coding", "git", was_referenced=True)
        ranked = opt.rank_context("coding", ["git", "clipboard"])
        assert ranked[0] == "git"

    def test_cache_stats(self):
        opt = self._make_optimizer()
        stats = opt.cache_stats()
        assert "entries" in stats
        assert "hits" in stats
