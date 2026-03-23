# src/homie_core/meta_learning/strategy_selector.py
"""Strategy Selector — learns which strategies work best for different task types."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ── default strategies per broad task category ──────────────────────────
_DEFAULT_STRATEGIES: dict[str, list[dict]] = {
    "code_generation": [
        {"agents": ["coder"], "planning": "chain_of_thought", "tools": ["editor"]},
        {"agents": ["coder", "reviewer"], "planning": "plan_and_execute", "tools": ["editor", "linter"]},
    ],
    "research": [
        {"agents": ["researcher"], "planning": "breadth_first", "tools": ["web_search"]},
        {"agents": ["researcher", "summariser"], "planning": "iterative_deepening", "tools": ["web_search", "rag"]},
    ],
    "conversation": [
        {"agents": ["conversationalist"], "planning": "reactive", "tools": []},
    ],
}

EXPLORE_RATE = 0.15  # epsilon for epsilon-greedy exploration


@dataclass
class _StrategyRecord:
    """Accumulated performance data for one (task_type, strategy_key) pair."""

    attempts: int = 0
    successes: int = 0
    total_duration_ms: float = 0.0
    total_quality: float = 0.0
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.attempts if self.attempts else 0.0

    @property
    def avg_quality(self) -> float:
        return self.total_quality / self.attempts if self.attempts else 0.0

    def score(self) -> float:
        """Composite score used for ranking (higher is better)."""
        return 0.6 * self.success_rate + 0.4 * self.avg_quality


def _strategy_key(strategy: dict) -> str:
    """Deterministic string key for a strategy dict."""
    agents = ",".join(sorted(strategy.get("agents", [])))
    planning = strategy.get("planning", "")
    tools = ",".join(sorted(strategy.get("tools", [])))
    return f"{agents}|{planning}|{tools}"


class StrategySelector:
    """Learns which strategies work best for different task types."""

    def __init__(self, storage: Any | None = None, explore_rate: float = EXPLORE_RATE):
        self._storage = storage
        self._explore_rate = explore_rate
        # {task_type: {strategy_key: _StrategyRecord}}
        self._records: dict[str, dict[str, _StrategyRecord]] = {}
        # {task_type: {strategy_key: strategy_dict}}
        self._strategies: dict[str, dict[str, dict]] = {}
        self._load_defaults()

    # ── public API ──────────────────────────────────────────────────────

    def select_strategy(self, task_type: str, context: dict | None = None) -> dict:
        """Select optimal strategy using epsilon-greedy over historical data."""
        context = context or {}
        candidates = self._strategies.get(task_type, {})

        if not candidates:
            log.info("No strategies registered for %s — returning empty.", task_type)
            return {"agents": [], "planning": "reactive", "tools": []}

        # Explore: pick randomly
        if random.random() < self._explore_rate:
            key = random.choice(list(candidates.keys()))
            log.debug("Exploring strategy %s for %s", key, task_type)
            return dict(candidates[key])

        # Exploit: pick the highest-scoring strategy
        best_key, best_score = None, -1.0
        for key in candidates:
            record = self._records.get(task_type, {}).get(key)
            score = record.score() if record else 0.0
            if score > best_score:
                best_score = score
                best_key = key

        chosen = candidates[best_key]  # type: ignore[index]
        log.debug("Selected strategy %s (score=%.3f) for %s", best_key, best_score, task_type)
        return dict(chosen)

    def record_outcome(
        self,
        task_type: str,
        strategy: dict,
        success: bool,
        metrics: dict | None = None,
    ) -> None:
        """Record how well a strategy worked."""
        metrics = metrics or {}
        key = _strategy_key(strategy)

        if task_type not in self._records:
            self._records[task_type] = {}
        if key not in self._records[task_type]:
            self._records[task_type][key] = _StrategyRecord()

        rec = self._records[task_type][key]
        rec.attempts += 1
        if success:
            rec.successes += 1
        rec.total_duration_ms += metrics.get("duration_ms", 0.0)
        rec.total_quality += metrics.get("quality", 0.0)
        rec.last_used = time.time()

        # Ensure strategy is registered
        self._strategies.setdefault(task_type, {})[key] = strategy

    def get_strategy_stats(self, task_type: str) -> dict:
        """Get performance stats per strategy for a task type."""
        records = self._records.get(task_type, {})
        return {
            key: {
                "attempts": r.attempts,
                "success_rate": round(r.success_rate, 4),
                "avg_duration_ms": round(r.avg_duration_ms, 2),
                "avg_quality": round(r.avg_quality, 4),
                "score": round(r.score(), 4),
            }
            for key, r in records.items()
        }

    def register_strategy(self, task_type: str, strategy: dict) -> None:
        """Register a new strategy for a task type."""
        key = _strategy_key(strategy)
        self._strategies.setdefault(task_type, {})[key] = strategy

    # ── internals ───────────────────────────────────────────────────────

    def _load_defaults(self) -> None:
        for task_type, strategies in _DEFAULT_STRATEGIES.items():
            for s in strategies:
                self.register_strategy(task_type, s)
