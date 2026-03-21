"""Improvement engine — observes, diagnoses, prescribes, and applies self-improvements."""

import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Optional

from ..event_bus import EventBus, HealthEvent
from ..health_log import HealthLog
from ..metrics import MetricsCollector
from .rollback import RollbackManager

logger = logging.getLogger(__name__)


class ImprovementLevel(IntEnum):
    CONFIG = 1
    WORKFLOW = 2
    CODE_PATCH = 3
    ARCHITECTURE = 4


@dataclass
class Observation:
    """An observed performance issue or optimization opportunity."""

    module: str
    metric: str
    current_value: float
    baseline_value: float
    description: str


class ImprovementEngine:
    """The self-improvement loop: observe -> diagnose -> prescribe -> apply -> monitor."""

    def __init__(
        self,
        event_bus: EventBus,
        health_log: HealthLog,
        metrics: MetricsCollector,
        rollback_manager: RollbackManager,
        inference_fn: Callable[[str], str],
        max_mutations_per_day: int = 10,
        monitoring_window: float = 300.0,
        error_threshold: float = 0.20,
        latency_threshold: float = 0.50,
    ) -> None:
        self._bus = event_bus
        self._log = health_log
        self._metrics = metrics
        self._rollback = rollback_manager
        self._infer = inference_fn
        self._max_mutations = max_mutations_per_day
        self._monitoring_window = monitoring_window
        self._error_threshold = error_threshold
        self._latency_threshold = latency_threshold

        self._mutations_today: int = 0
        self._last_reset_day: int = 0
        self._core_locks: list[str] = []

    def analyze(self) -> list[Observation]:
        """Step 1: Observe — analyze current metrics for optimization opportunities."""
        observations = []
        snapshot = self._metrics.snapshot()

        for module, metrics in snapshot.items():
            for metric_name, values in metrics.items():
                latest = values.get("latest", 0)
                average = values.get("average", 0)
                count = values.get("count", 0)

                if count < 10:
                    continue

                # Flag if latest is significantly above average
                if average > 0 and latest > average * 1.5:
                    observations.append(Observation(
                        module=module,
                        metric=metric_name,
                        current_value=latest,
                        baseline_value=average,
                        description=f"{module}.{metric_name} is {latest:.1f} vs baseline {average:.1f}",
                    ))

        return observations

    def can_mutate(self) -> bool:
        """Check if mutation rate limit allows another change."""
        today = int(time.time() / 86400)
        if today != self._last_reset_day:
            self._mutations_today = 0
            self._last_reset_day = today
        return self._mutations_today < self._max_mutations

    def record_mutation(self) -> None:
        """Record that a mutation was applied."""
        today = int(time.time() / 86400)
        if today != self._last_reset_day:
            self._mutations_today = 0
            self._last_reset_day = today
        self._mutations_today += 1

    def add_core_lock(self, path: str) -> None:
        """Add a path to the core lock list (immutable files)."""
        self._core_locks.append(path)

    def is_locked(self, path: str) -> bool:
        """Check if a file path is core-locked."""
        for lock in self._core_locks:
            if lock.endswith("/"):
                if path.startswith(lock) or path.startswith(lock.rstrip("/")):
                    return True
            elif path == lock:
                return True
        return False
