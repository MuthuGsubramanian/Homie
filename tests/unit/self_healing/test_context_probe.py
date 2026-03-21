# tests/unit/self_healing/test_context_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.context_probe import ContextProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestContextProbe:
    def test_healthy_when_aggregator_works(self):
        agg = MagicMock()
        agg.tick.return_value = {"active_window": "VSCode", "timestamp": 1.0}
        probe = ContextProbe(context_aggregator=agg)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_tick_returns_empty(self):
        agg = MagicMock()
        agg.tick.return_value = {}
        probe = ContextProbe(context_aggregator=agg)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_tick_raises(self):
        agg = MagicMock()
        agg.tick.side_effect = RuntimeError("observer crash")
        probe = ContextProbe(context_aggregator=agg)
        result = probe.check()
        assert result.status == HealthStatus.FAILED

    def test_handles_none_aggregator(self):
        probe = ContextProbe(context_aggregator=None)
        result = probe.check()
        assert result.status == HealthStatus.UNKNOWN
