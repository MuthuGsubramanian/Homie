# tests/unit/self_healing/test_improvement_engine.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.improvement.engine import ImprovementEngine, ImprovementLevel


class TestImprovementLevel:
    def test_ordering(self):
        assert ImprovementLevel.CONFIG < ImprovementLevel.WORKFLOW
        assert ImprovementLevel.WORKFLOW < ImprovementLevel.CODE_PATCH
        assert ImprovementLevel.CODE_PATCH < ImprovementLevel.ARCHITECTURE


class TestImprovementEngine:
    def _make_engine(self, **overrides):
        defaults = {
            "event_bus": MagicMock(),
            "health_log": MagicMock(),
            "metrics": MagicMock(),
            "rollback_manager": MagicMock(),
            "inference_fn": MagicMock(return_value="no improvements needed"),
            "max_mutations_per_day": 10,
            "monitoring_window": 1,
            "error_threshold": 0.20,
            "latency_threshold": 0.50,
        }
        defaults.update(overrides)
        return ImprovementEngine(**defaults)

    def test_analyze_returns_observations(self):
        metrics = MagicMock()
        metrics.snapshot.return_value = {
            "inference": {"latency_ms": {"latest": 500, "average": 100, "count": 50}}
        }
        engine = self._make_engine(metrics=metrics)
        observations = engine.analyze()
        assert len(observations) > 0

    def test_rate_limit_respected(self):
        engine = self._make_engine(max_mutations_per_day=0)
        assert engine.can_mutate() is False

    def test_rate_limit_allows_when_under(self):
        engine = self._make_engine(max_mutations_per_day=10)
        assert engine.can_mutate() is True

    def test_core_lock_prevents_modification(self):
        engine = self._make_engine()
        engine.add_core_lock("src/homie_core/self_healing/improvement/rollback.py")
        assert engine.is_locked("src/homie_core/self_healing/improvement/rollback.py")
        assert not engine.is_locked("src/homie_core/some_other.py")

    def test_core_lock_directory(self):
        engine = self._make_engine()
        engine.add_core_lock("src/homie_core/security/")
        assert engine.is_locked("src/homie_core/security/vault.py")
        assert not engine.is_locked("src/homie_core/storage/database.py")

    def test_mutation_count_increments(self):
        engine = self._make_engine(max_mutations_per_day=10)
        engine._mutations_today = 0
        engine.record_mutation()
        assert engine._mutations_today == 1
