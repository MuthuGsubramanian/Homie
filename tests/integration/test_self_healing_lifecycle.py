"""Integration test: full self-healing lifecycle from probe to recovery."""
import time
import pytest
from unittest.mock import MagicMock

from homie_core.self_healing.watchdog import HealthWatchdog
from homie_core.self_healing.probes.base import BaseProbe, HealthStatus, ProbeResult
from homie_core.self_healing.recovery.engine import RecoveryEngine, RecoveryResult, RecoveryTier


class FlakeyProbe(BaseProbe):
    """Probe that fails N times then succeeds."""
    name = "flakey"
    interval = 0.1

    def __init__(self, fail_count=2):
        self._calls = 0
        self._fail_count = fail_count

    def check(self):
        self._calls += 1
        if self._calls <= self._fail_count:
            return ProbeResult(status=HealthStatus.FAILED, latency_ms=1.0, error_count=1, last_error="simulated failure")
        return ProbeResult(status=HealthStatus.HEALTHY, latency_ms=1.0, error_count=0)


class TestSelfHealingLifecycle:
    def test_probe_failure_triggers_recovery(self, tmp_path):
        """Full flow: probe detects failure -> recovery engine attempts fix."""
        wd = HealthWatchdog(db_path=tmp_path / "test.db")

        probe = FlakeyProbe(fail_count=1)
        wd.register_probe(probe)

        # Set up recovery engine
        recovery = RecoveryEngine(
            event_bus=wd.event_bus,
            health_log=wd.health_log,
        )
        fixed = {"called": False}

        def fix_strategy(module, status, error, **ctx):
            fixed["called"] = True
            return RecoveryResult(success=True, action="fixed it", tier=RecoveryTier.RETRY)

        recovery.register_strategy("flakey", RecoveryTier.RETRY, fix_strategy)
        wd.set_recovery_engine(recovery)

        # First probe run — should fail and trigger recovery
        results = wd.run_all_probes()
        assert results["flakey"].status == HealthStatus.FAILED
        assert fixed["called"] is True

        # Second run — should succeed
        results = wd.run_all_probes()
        assert results["flakey"].status == HealthStatus.HEALTHY

        wd.stop()

    def test_metrics_tracked_across_probes(self, tmp_path):
        """Metrics are collected for each probe run."""
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        probe = FlakeyProbe(fail_count=0)  # always healthy
        wd.register_probe(probe)

        for _ in range(5):
            wd.run_all_probes()

        avg = wd.metrics.get_average("flakey", "latency_ms")
        assert avg is not None
        assert avg > 0

        wd.stop()

    def test_health_log_persists_events(self, tmp_path):
        """Health events are written to SQLite."""
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        probe = FlakeyProbe(fail_count=0)
        wd.register_probe(probe)

        wd.run_all_probes()

        events = wd.health_log.query(module="flakey")
        assert len(events) >= 1

        wd.stop()
