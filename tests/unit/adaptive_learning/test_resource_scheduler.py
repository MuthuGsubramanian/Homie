# tests/unit/adaptive_learning/test_resource_scheduler.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.performance.resource_scheduler import ResourceScheduler


class TestResourceScheduler:
    def test_record_activity(self):
        sched = ResourceScheduler()
        sched.record_activity(hour=9, activity="inference")
        sched.record_activity(hour=9, activity="inference")
        sched.record_activity(hour=22, activity="idle")
        pattern = sched.get_hour_pattern(9)
        assert pattern["inference"] == 2

    def test_predict_activity(self):
        sched = ResourceScheduler()
        for _ in range(10):
            sched.record_activity(hour=9, activity="inference")
        for _ in range(2):
            sched.record_activity(hour=9, activity="idle")
        prediction = sched.predict_activity(hour=9)
        assert prediction == "inference"

    def test_predict_returns_idle_for_unknown_hour(self):
        sched = ResourceScheduler()
        assert sched.predict_activity(hour=3) == "idle"

    def test_should_preload_during_active_hours(self):
        sched = ResourceScheduler()
        for _ in range(10):
            sched.record_activity(hour=9, activity="inference")
        assert sched.should_preload(hour=9) is True

    def test_should_not_preload_during_idle_hours(self):
        sched = ResourceScheduler()
        for _ in range(10):
            sched.record_activity(hour=3, activity="idle")
        assert sched.should_preload(hour=3) is False

    def test_get_schedule_summary(self):
        sched = ResourceScheduler()
        sched.record_activity(hour=9, activity="inference")
        summary = sched.get_schedule_summary()
        assert isinstance(summary, dict)
        assert 9 in summary
