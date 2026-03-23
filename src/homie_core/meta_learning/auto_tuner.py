# src/homie_core/meta_learning/auto_tuner.py
"""Auto Tuner — automatically tunes Homie's hyperparameters based on performance."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .performance_tracker import MetaPerformanceTracker

log = logging.getLogger(__name__)


@dataclass
class _TuningRecord:
    """History entry for an applied tuning."""

    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    applied_at: float
    reverted: bool = False


# ── tuning rules ────────────────────────────────────────────────────────
# Each rule is a callable(config, tracker) -> list[dict] | None

def _rule_cache_size(config: dict, tracker: MetaPerformanceTracker) -> list[dict]:
    """Suggest increasing cache if hit-rate is high (indicates cache is useful)."""
    suggestions: list[dict] = []
    cache_hit_rate = config.get("cache_hit_rate", 0.0)
    cache_max = config.get("cache_max_entries", 500)
    if cache_hit_rate >= 0.80 and cache_max < 5000:
        suggestions.append(
            {
                "parameter": "cache_max_entries",
                "old_value": cache_max,
                "new_value": min(cache_max * 2, 5000),
                "reason": f"Cache hit rate is {cache_hit_rate:.0%} — increasing capacity should help.",
            }
        )
    return suggestions


def _rule_probe_interval(config: dict, tracker: MetaPerformanceTracker) -> list[dict]:
    """Relax probe interval when the system is healthy for a while."""
    suggestions: list[dict] = []
    health = tracker.get_overall_health()
    probe_interval = config.get("probe_interval_s", 30)
    if health.get("status") == "healthy" and probe_interval < 120:
        suggestions.append(
            {
                "parameter": "probe_interval_s",
                "old_value": probe_interval,
                "new_value": min(probe_interval * 2, 120),
                "reason": "System is healthy — reducing probe frequency to save resources.",
            }
        )
    return suggestions


def _rule_explore_rate(config: dict, tracker: MetaPerformanceTracker) -> list[dict]:
    """Lower exploration when overall quality is high, raise it when struggling."""
    suggestions: list[dict] = []
    health = tracker.get_overall_health()
    explore = config.get("explore_rate", 0.15)
    sr = health.get("overall_success_rate", 0.0)

    if sr >= 0.9 and explore > 0.05:
        suggestions.append(
            {
                "parameter": "explore_rate",
                "old_value": explore,
                "new_value": max(explore * 0.5, 0.05),
                "reason": f"Success rate is {sr:.0%} — reducing exploration to exploit winning strategies.",
            }
        )
    elif sr < 0.6 and explore < 0.30:
        suggestions.append(
            {
                "parameter": "explore_rate",
                "old_value": explore,
                "new_value": min(explore * 2, 0.30),
                "reason": f"Success rate is {sr:.0%} — increasing exploration to find better strategies.",
            }
        )
    return suggestions


_RULES = [_rule_cache_size, _rule_probe_interval, _rule_explore_rate]


class AutoTuner:
    """Automatically tunes Homie's hyperparameters."""

    def __init__(self, config: dict, performance_tracker: MetaPerformanceTracker):
        self._config = config
        self._tracker = performance_tracker
        self._history: list[_TuningRecord] = []

    # ── public API ──────────────────────────────────────────────────────

    def suggest_tunings(self) -> list[dict]:
        """Suggest parameter changes based on current config and performance data."""
        suggestions: list[dict] = []
        for rule in _RULES:
            try:
                result = rule(self._config, self._tracker)
                if result:
                    suggestions.extend(result)
            except Exception:
                log.warning("Tuning rule %s failed", rule.__name__, exc_info=True)
        return suggestions

    def apply_tuning(self, tuning: dict) -> bool:
        """Apply a suggested tuning to the live config. Returns True on success."""
        param = tuning.get("parameter")
        new_val = tuning.get("new_value")
        if not param:
            return False

        old_val = self._config.get(param)
        self._config[param] = new_val

        self._history.append(
            _TuningRecord(
                parameter=param,
                old_value=old_val,
                new_value=new_val,
                reason=tuning.get("reason", ""),
                applied_at=time.time(),
            )
        )
        log.info("Applied tuning: %s %s -> %s (%s)", param, old_val, new_val, tuning.get("reason", ""))
        return True

    def revert_tuning(self, parameter: str) -> bool:
        """Revert the most recent tuning for *parameter*."""
        for rec in reversed(self._history):
            if rec.parameter == parameter and not rec.reverted:
                self._config[parameter] = rec.old_value
                rec.reverted = True
                log.info("Reverted tuning: %s back to %s", parameter, rec.old_value)
                return True
        return False

    def get_tuning_history(self) -> list[dict]:
        """History of applied tunings and their outcomes."""
        return [
            {
                "parameter": r.parameter,
                "old_value": r.old_value,
                "new_value": r.new_value,
                "reason": r.reason,
                "applied_at": r.applied_at,
                "reverted": r.reverted,
            }
            for r in self._history
        ]
