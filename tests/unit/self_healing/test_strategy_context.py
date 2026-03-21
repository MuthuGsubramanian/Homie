# tests/unit/self_healing/test_strategy_context.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.recovery.strategies.context import (
    restart_observer,
    reduce_monitoring_frequency,
    degrade_without_context,
)
from homie_core.self_healing.recovery.engine import RecoveryResult, RecoveryTier


class TestContextRecoveryStrategies:
    def test_restart_observer_success(self):
        agg = MagicMock()
        agg.tick.return_value = {"active_window": "test"}
        result = restart_observer(module="context", status=2, error="crash", context_aggregator=agg)
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY

    def test_restart_observer_failure(self):
        agg = MagicMock()
        agg.tick.side_effect = RuntimeError("still broken")
        result = restart_observer(module="context", status=2, error="crash", context_aggregator=agg)
        assert result.success is False

    def test_reduce_monitoring_frequency(self):
        config = MagicMock()
        result = reduce_monitoring_frequency(module="context", status=2, error="overloaded", config=config)
        assert result.success is True
        assert result.tier == RecoveryTier.FALLBACK

    def test_degrade_without_context(self):
        result = degrade_without_context(module="context", status=2, error="fatal")
        assert result.success is True
        assert result.tier == RecoveryTier.DEGRADE
