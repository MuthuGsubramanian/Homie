# tests/unit/meta_learning/test_auto_tuner.py
"""Tests for the AutoTuner."""

import pytest

from homie_core.meta_learning.auto_tuner import AutoTuner
from homie_core.meta_learning.performance_tracker import MetaPerformanceTracker


class TestAutoTuner:
    def _make_tuner(self, config=None, tracker=None):
        config = config or {}
        tracker = tracker or MetaPerformanceTracker()
        return AutoTuner(config, tracker), config, tracker

    def test_suggest_cache_increase(self):
        tuner, cfg, _ = self._make_tuner({"cache_hit_rate": 0.90, "cache_max_entries": 500})
        suggestions = tuner.suggest_tunings()
        cache_sug = [s for s in suggestions if s["parameter"] == "cache_max_entries"]
        assert len(cache_sug) == 1
        assert cache_sug[0]["new_value"] == 1000

    def test_no_cache_suggestion_when_hit_rate_low(self):
        tuner, _, _ = self._make_tuner({"cache_hit_rate": 0.50, "cache_max_entries": 500})
        suggestions = tuner.suggest_tunings()
        cache_sug = [s for s in suggestions if s["parameter"] == "cache_max_entries"]
        assert len(cache_sug) == 0

    def test_suggest_probe_interval_increase_when_healthy(self):
        tracker = MetaPerformanceTracker()
        for _ in range(20):
            tracker.record_task("code", 100, True, 0.9)
        tuner, _, _ = self._make_tuner({"probe_interval_s": 30}, tracker)
        suggestions = tuner.suggest_tunings()
        probe_sug = [s for s in suggestions if s["parameter"] == "probe_interval_s"]
        assert len(probe_sug) == 1
        assert probe_sug[0]["new_value"] == 60

    def test_suggest_explore_rate_decrease_when_winning(self):
        tracker = MetaPerformanceTracker()
        for _ in range(20):
            tracker.record_task("code", 50, True, 0.95)
        tuner, _, _ = self._make_tuner({"explore_rate": 0.15}, tracker)
        suggestions = tuner.suggest_tunings()
        explore_sug = [s for s in suggestions if s["parameter"] == "explore_rate"]
        assert len(explore_sug) == 1
        assert explore_sug[0]["new_value"] < 0.15

    def test_suggest_explore_rate_increase_when_struggling(self):
        tracker = MetaPerformanceTracker()
        for _ in range(20):
            tracker.record_task("code", 500, False, 0.2)
        tuner, _, _ = self._make_tuner({"explore_rate": 0.10}, tracker)
        suggestions = tuner.suggest_tunings()
        explore_sug = [s for s in suggestions if s["parameter"] == "explore_rate"]
        assert len(explore_sug) == 1
        assert explore_sug[0]["new_value"] > 0.10

    def test_apply_tuning_updates_config(self):
        tuner, cfg, _ = self._make_tuner({"cache_max_entries": 500})
        result = tuner.apply_tuning({"parameter": "cache_max_entries", "new_value": 1000, "reason": "test"})
        assert result is True
        assert cfg["cache_max_entries"] == 1000

    def test_apply_tuning_records_history(self):
        tuner, _, _ = self._make_tuner({"x": 1})
        tuner.apply_tuning({"parameter": "x", "new_value": 2, "reason": "testing"})
        history = tuner.get_tuning_history()
        assert len(history) == 1
        assert history[0]["parameter"] == "x"
        assert history[0]["old_value"] == 1
        assert history[0]["new_value"] == 2

    def test_revert_tuning(self):
        tuner, cfg, _ = self._make_tuner({"x": 10})
        tuner.apply_tuning({"parameter": "x", "new_value": 20, "reason": "test"})
        assert cfg["x"] == 20
        reverted = tuner.revert_tuning("x")
        assert reverted is True
        assert cfg["x"] == 10

    def test_revert_nonexistent_returns_false(self):
        tuner, _, _ = self._make_tuner()
        assert tuner.revert_tuning("nonexistent") is False
