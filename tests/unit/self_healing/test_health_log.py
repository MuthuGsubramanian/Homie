# tests/unit/self_healing/test_health_log.py
import json
import pytest
from homie_core.self_healing.health_log import HealthLog
from homie_core.self_healing.event_bus import HealthEvent


class TestHealthLog:
    def test_write_and_read_event(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        evt = HealthEvent(module="inference", event_type="probe_result", severity="info", details={"latency_ms": 42})
        log.write(evt)
        events = log.query(module="inference", limit=10)
        assert len(events) == 1
        assert events[0]["module"] == "inference"
        assert events[0]["severity"] == "info"
        assert json.loads(events[0]["details"])["latency_ms"] == 42

    def test_query_by_event_type(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        log.write(HealthEvent(module="m", event_type="probe_result", severity="info", details={}))
        log.write(HealthEvent(module="m", event_type="recovery", severity="warning", details={}))
        log.write(HealthEvent(module="m", event_type="probe_result", severity="info", details={}))
        results = log.query(event_type="recovery")
        assert len(results) == 1

    def test_query_by_severity(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        log.write(HealthEvent(module="m", event_type="t", severity="info", details={}))
        log.write(HealthEvent(module="m", event_type="t", severity="error", details={}))
        results = log.query(min_severity="error")
        assert len(results) == 1

    def test_query_limit(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        for i in range(10):
            log.write(HealthEvent(module="m", event_type="t", severity="info", details={"i": i}))
        results = log.query(limit=3)
        assert len(results) == 3

    def test_cleanup_old_events(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        # Write event with old timestamp
        log.write(HealthEvent(module="m", event_type="t", severity="info", details={}, timestamp=1.0))
        log.write(HealthEvent(module="m", event_type="t", severity="info", details={}))
        deleted = log.cleanup(max_age_days=1)
        assert deleted == 1
        assert len(log.query()) == 1

    def test_close(self, tmp_path):
        log = HealthLog(db_path=tmp_path / "test.db")
        log.initialize()
        log.close()
        # Should not raise on double close
        log.close()
