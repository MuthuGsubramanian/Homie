# tests/unit/meta_learning/test_strategy_selector.py
"""Tests for the StrategySelector."""

import pytest

from homie_core.meta_learning.strategy_selector import StrategySelector, _strategy_key


class TestStrategySelector:
    def _make_selector(self, explore_rate: float = 0.0):
        """Create a selector with exploration disabled for deterministic tests."""
        return StrategySelector(explore_rate=explore_rate)

    def test_default_strategies_loaded(self):
        sel = self._make_selector()
        # code_generation should have default strategies registered
        strat = sel.select_strategy("code_generation")
        assert "agents" in strat
        assert "planning" in strat

    def test_select_returns_empty_for_unknown_task(self):
        sel = self._make_selector()
        strat = sel.select_strategy("completely_unknown_task")
        assert strat["agents"] == []
        assert strat["planning"] == "reactive"

    def test_record_and_exploit_best(self):
        sel = self._make_selector()
        good = {"agents": ["a"], "planning": "x", "tools": []}
        bad = {"agents": ["b"], "planning": "y", "tools": []}
        sel.register_strategy("test_task", good)
        sel.register_strategy("test_task", bad)

        # good succeeds 9/10, bad succeeds 1/10
        for _ in range(9):
            sel.record_outcome("test_task", good, True, {"quality": 0.9})
        sel.record_outcome("test_task", good, False, {"quality": 0.2})

        sel.record_outcome("test_task", bad, True, {"quality": 0.3})
        for _ in range(9):
            sel.record_outcome("test_task", bad, False, {"quality": 0.1})

        # With explore=0, should always pick the good one
        chosen = sel.select_strategy("test_task")
        assert chosen["agents"] == ["a"]

    def test_record_outcome_creates_record(self):
        sel = self._make_selector()
        strat = {"agents": ["x"], "planning": "p", "tools": ["t"]}
        sel.record_outcome("new_type", strat, True, {"duration_ms": 100, "quality": 0.8})
        stats = sel.get_strategy_stats("new_type")
        assert len(stats) == 1
        key = list(stats.keys())[0]
        assert stats[key]["attempts"] == 1
        assert stats[key]["success_rate"] == 1.0

    def test_get_strategy_stats_empty(self):
        sel = self._make_selector()
        assert sel.get_strategy_stats("nonexistent") == {}

    def test_strategy_key_deterministic(self):
        s = {"agents": ["b", "a"], "planning": "p", "tools": ["z", "m"]}
        k1 = _strategy_key(s)
        k2 = _strategy_key(s)
        assert k1 == k2
        # Order should not matter
        s2 = {"agents": ["a", "b"], "planning": "p", "tools": ["m", "z"]}
        assert _strategy_key(s2) == k1

    def test_exploration_returns_valid_strategy(self):
        sel = self._make_selector(explore_rate=1.0)  # always explore
        strat = sel.select_strategy("code_generation")
        assert "agents" in strat
        assert "planning" in strat

    def test_register_custom_strategy(self):
        sel = self._make_selector()
        custom = {"agents": ["custom_agent"], "planning": "custom_plan", "tools": ["custom_tool"]}
        sel.register_strategy("my_task", custom)
        chosen = sel.select_strategy("my_task")
        assert chosen["agents"] == ["custom_agent"]
