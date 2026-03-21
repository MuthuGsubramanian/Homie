# tests/unit/self_healing/test_preemptive.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.recovery.preemptive import PreemptiveEngine, PreemptiveRule


class TestPreemptiveRule:
    def test_rule_creation(self):
        rule = PreemptiveRule(
            name="gpu_memory_guard",
            module="inference",
            condition_metric="gpu_mem_percent",
            threshold=85.0,
            action="reduce gpu_layers",
            observation_count=0,
            min_observations=3,
        )
        assert rule.is_active is False  # needs 3 observations

    def test_rule_activates_after_threshold(self):
        rule = PreemptiveRule(
            name="test",
            module="m",
            condition_metric="v",
            threshold=50.0,
            action="act",
            observation_count=3,
            min_observations=3,
        )
        assert rule.is_active is True


class TestPreemptiveEngine:
    def test_add_and_list_rules(self):
        pe = PreemptiveEngine(metrics=MagicMock())
        pe.add_rule(PreemptiveRule(
            name="test_rule",
            module="inference",
            condition_metric="latency_ms",
            threshold=500.0,
            action="reduce context",
            observation_count=3,
            min_observations=3,
        ))
        assert len(pe.rules) == 1

    def test_evaluate_triggers_rule(self):
        metrics = MagicMock()
        metrics.get_latest.return_value = 600.0  # above threshold
        pe = PreemptiveEngine(metrics=metrics)
        rule = PreemptiveRule(
            name="high_latency",
            module="inference",
            condition_metric="latency_ms",
            threshold=500.0,
            action="reduce context",
            observation_count=5,
            min_observations=3,
        )
        pe.add_rule(rule)
        triggered = pe.evaluate()
        assert len(triggered) == 1
        assert triggered[0].name == "high_latency"

    def test_evaluate_ignores_inactive_rules(self):
        metrics = MagicMock()
        metrics.get_latest.return_value = 600.0
        pe = PreemptiveEngine(metrics=metrics)
        rule = PreemptiveRule(
            name="inactive",
            module="inference",
            condition_metric="latency_ms",
            threshold=500.0,
            action="act",
            observation_count=1,  # below min_observations
            min_observations=3,
        )
        pe.add_rule(rule)
        triggered = pe.evaluate()
        assert len(triggered) == 0

    def test_evaluate_ignores_below_threshold(self):
        metrics = MagicMock()
        metrics.get_latest.return_value = 100.0  # below threshold
        pe = PreemptiveEngine(metrics=metrics)
        rule = PreemptiveRule(
            name="ok",
            module="inference",
            condition_metric="latency_ms",
            threshold=500.0,
            action="act",
            observation_count=5,
            min_observations=3,
        )
        pe.add_rule(rule)
        triggered = pe.evaluate()
        assert len(triggered) == 0

    def test_seed_rules_exist(self):
        pe = PreemptiveEngine(metrics=MagicMock(), seed=True)
        assert len(pe.rules) > 0
