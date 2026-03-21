# tests/unit/self_healing/test_watchdog.py
import time
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.watchdog import HealthWatchdog
from homie_core.self_healing.probes.base import BaseProbe, HealthStatus, ProbeResult


class FakeProbe(BaseProbe):
    name = "fake"
    interval = 0.1

    def __init__(self, status=HealthStatus.HEALTHY):
        self._status = status

    def check(self):
        return ProbeResult(status=self._status, latency_ms=1.0, error_count=0)


class TestHealthWatchdog:
    def test_register_probe(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        probe = FakeProbe()
        wd.register_probe(probe)
        assert "fake" in wd.registered_probes

    def test_run_all_probes(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        wd.register_probe(FakeProbe(HealthStatus.HEALTHY))
        results = wd.run_all_probes()
        assert "fake" in results
        assert results["fake"].status == HealthStatus.HEALTHY

    def test_system_health_all_healthy(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        wd.register_probe(FakeProbe(HealthStatus.HEALTHY))
        wd.run_all_probes()
        assert wd.system_health == HealthStatus.HEALTHY

    def test_system_health_worst_wins(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        healthy = FakeProbe(HealthStatus.HEALTHY)
        healthy.name = "good"
        failed = FakeProbe(HealthStatus.FAILED)
        failed.name = "bad"
        wd.register_probe(healthy)
        wd.register_probe(failed)
        wd.run_all_probes()
        assert wd.system_health == HealthStatus.FAILED

    def test_start_and_stop(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        wd.register_probe(FakeProbe())
        wd.start()
        time.sleep(0.3)  # let it run a few cycles
        wd.stop()
        # Should not hang
        assert wd._running is False

    def test_recovery_triggered_on_failure(self, tmp_path):
        wd = HealthWatchdog(db_path=tmp_path / "test.db")
        failed = FakeProbe(HealthStatus.FAILED)
        failed.name = "failing"
        wd.register_probe(failed)

        recovery = MagicMock()
        wd.set_recovery_engine(recovery)
        wd.run_all_probes()
        # Recovery should be attempted for the failed probe
        recovery.recover.assert_called_once()
