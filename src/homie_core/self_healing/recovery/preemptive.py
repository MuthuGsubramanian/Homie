"""Preemptive healing — fix problems before they happen."""

import logging
from dataclasses import dataclass
from typing import Optional

from ..metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class PreemptiveRule:
    """A rule that triggers preemptive action based on metric conditions."""

    name: str
    module: str
    condition_metric: str
    threshold: float
    action: str
    observation_count: int = 0
    min_observations: int = 3

    @property
    def is_active(self) -> bool:
        return self.observation_count >= self.min_observations


def _seed_rules() -> list[PreemptiveRule]:
    """Seed preemptive rules — the starting set that evolves over time."""
    return [
        PreemptiveRule(
            name="gpu_memory_guard",
            module="inference",
            condition_metric="gpu_mem_percent",
            threshold=90.0,
            action="reduce gpu_layers proactively",
            observation_count=0,
            min_observations=3,
        ),
        PreemptiveRule(
            name="inference_latency_guard",
            module="inference",
            condition_metric="latency_ms",
            threshold=5000.0,
            action="clear caches and optimize settings",
            observation_count=0,
            min_observations=3,
        ),
        PreemptiveRule(
            name="disk_space_guard",
            module="storage",
            condition_metric="disk_usage_percent",
            threshold=90.0,
            action="trigger early cleanup cycle",
            observation_count=0,
            min_observations=3,
        ),
    ]


class PreemptiveEngine:
    """Evaluates preemptive rules against current metrics."""

    def __init__(
        self,
        metrics: MetricsCollector,
        seed: bool = False,
    ) -> None:
        self._metrics = metrics
        self._rules: list[PreemptiveRule] = _seed_rules() if seed else []

    @property
    def rules(self) -> list[PreemptiveRule]:
        return list(self._rules)

    def add_rule(self, rule: PreemptiveRule) -> None:
        """Add a preemptive rule."""
        self._rules.append(rule)

    def evaluate(self) -> list[PreemptiveRule]:
        """Evaluate all active rules against current metrics. Returns triggered rules."""
        triggered = []
        for rule in self._rules:
            if not rule.is_active:
                continue

            value = self._metrics.get_latest(rule.module, rule.condition_metric)
            if value is None:
                continue

            if value >= rule.threshold:
                logger.info(
                    "Preemptive rule '%s' triggered: %s=%s (threshold=%s)",
                    rule.name, rule.condition_metric, value, rule.threshold,
                )
                triggered.append(rule)

        return triggered
