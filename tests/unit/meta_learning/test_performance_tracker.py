# tests/unit/meta_learning/test_performance_tracker.py
"""Tests for the MetaPerformanceTracker."""

import time
from unittest.mock import patch

import pytest

from homie_core.meta_learning.performance_tracker import MetaPerformanceTracker


class TestMetaPerformanceTracker:
    def _make_tracker(self):
        return MetaPerformanceTracker()

    def test_record_task_stores_entry(self):
        t = self._make_tracker()
        t.record_task("code", 100.0, True, 0.9)
        assert len(t._entries) == 1
        assert t._entries[0].task_type == "code"

    def test_quality_score_clamped(self):
        t = self._make_tracker()
        t.record_task("x", 10, True, 1.5)
        t.record_task("x", 10, True, -0.5)
        assert t._entries[0].quality_score == 1.0
        assert t._entries[1].quality_score == 0.0

    def test_improvement_trend_insufficient_data(self):
        t = self._make_tracker()
        result = t.get_improvement_trend("code")
        assert result["direction"] == "insufficient_data"
        assert result["sample_size"] == 0

    def test_improvement_trend_improving(self):
        t = self._make_tracker()
        now = time.time()
        # First half: poor results (older)
        for i in range(10):
            entry = t._entries
            t.record_task("code", 200, False, 0.3)
            t._entries[-1].timestamp = now - 20 * 86400 + i
        # Second half: good results (recent)
        for i in range(10):
            t.record_task("code", 100, True, 0.9)
            t._entries[-1].timestamp = now - 5 * 86400 + i

        trend = t.get_improvement_trend("code", window_days=30)
        assert trend["direction"] == "improving"
        assert trend["improvement_rate"] > 0

    def test_improvement_trend_declining(self):
        t = self._make_tracker()
        now = time.time()
        for i in range(10):
            t.record_task("code", 100, True, 0.9)
            t._entries[-1].timestamp = now - 20 * 86400 + i
        for i in range(10):
            t.record_task("code", 500, False, 0.2)
            t._entries[-1].timestamp = now - 5 * 86400 + i

        trend = t.get_improvement_trend("code", window_days=30)
        assert trend["direction"] == "declining"

    def test_get_bottlenecks(self):
        t = self._make_tracker()
        # Good task type
        for _ in range(10):
            t.record_task("good_task", 50, True, 0.95)
        # Struggling task type
        for _ in range(10):
            t.record_task("bad_task", 500, False, 0.2)

        bottlenecks = t.get_bottlenecks()
        assert len(bottlenecks) == 1
        assert bottlenecks[0]["task_type"] == "bad_task"

    def test_get_overall_health_no_data(self):
        t = self._make_tracker()
        h = t.get_overall_health()
        assert h["status"] == "no_data"
        assert h["total_tasks"] == 0

    def test_get_overall_health_healthy(self):
        t = self._make_tracker()
        for _ in range(20):
            t.record_task("code", 100, True, 0.9)
        h = t.get_overall_health()
        assert h["status"] == "healthy"
        assert h["overall_success_rate"] == 1.0

    def test_get_overall_health_needs_attention(self):
        t = self._make_tracker()
        for _ in range(20):
            t.record_task("code", 500, False, 0.2)
        h = t.get_overall_health()
        assert h["status"] == "needs_attention"
