# tests/unit/self_healing/test_recovery_history.py
import pytest
from homie_core.self_healing.health_log import HealthLog


class TestRecoveryHistory:
    def test_write_and_query_recovery(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        log.write_recovery(
            module="inference",
            failure_type="timeout",
            tier=1,
            action="retry with reduced tokens",
            success=True,
            time_to_recover_ms=1200,
            system_state={"gpu_mem": "11.2GB"},
        )
        history = log.query_recovery(module="inference")
        assert len(history) == 1
        assert history[0]["failure_type"] == "timeout"
        assert history[0]["success"] == 1

    def test_query_recovery_by_failure_type(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        log.write_recovery(module="inference", failure_type="timeout", tier=1, action="a", success=True, time_to_recover_ms=100, system_state={})
        log.write_recovery(module="inference", failure_type="oom", tier=2, action="b", success=False, time_to_recover_ms=200, system_state={})
        results = log.query_recovery(module="inference", failure_type="oom")
        assert len(results) == 1
        assert results[0]["success"] == 0

    def test_recovery_pattern_summary(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        for _ in range(5):
            log.write_recovery(module="inference", failure_type="timeout", tier=1, action="retry", success=True, time_to_recover_ms=100, system_state={})
        for _ in range(2):
            log.write_recovery(module="inference", failure_type="timeout", tier=1, action="retry", success=False, time_to_recover_ms=100, system_state={})
        patterns = log.recovery_pattern_summary("inference", "timeout")
        assert patterns["total"] == 7
        assert patterns["success_count"] == 5
        assert patterns["success_rate"] == pytest.approx(5 / 7)
