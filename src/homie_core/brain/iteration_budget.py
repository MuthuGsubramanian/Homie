"""Iteration Budget — thread-safe shared counter for agentic loops.

Prevents runaway token consumption when the agentic loop spawns
sub-tasks or when tools trigger recursive agent calls. A single
budget instance is shared across parent and child agents.

Inspired by hermes-agent's IterationBudget pattern.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class IterationBudget:
    """Thread-safe shared iteration counter.

    Usage:
        budget = IterationBudget(max_iterations=20)
        while budget.consume():
            # do work
            pass

    Child agents share the same budget:
        child_budget = budget.child(reserved=5)
    """

    max_iterations: int = 20
    _used: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _parent: IterationBudget | None = field(default=None, repr=False)

    @property
    def remaining(self) -> int:
        """Iterations remaining in this budget."""
        with self._lock:
            return max(0, self.max_iterations - self._used)

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._used >= self.max_iterations

    def consume(self, n: int = 1) -> bool:
        """Consume N iterations. Returns True if budget was available.

        Also consumes from parent budget if this is a child.
        """
        with self._lock:
            if self._used + n > self.max_iterations:
                return False
            self._used += n

        # Propagate to parent
        if self._parent is not None:
            if not self._parent.consume(n):
                # Parent exhausted — roll back local consumption
                with self._lock:
                    self._used -= n
                return False

        return True

    def child(self, reserved: int | None = None) -> IterationBudget:
        """Create a child budget that shares this budget's counter.

        Args:
            reserved: Max iterations for the child. Defaults to
                      half of parent's remaining budget.
        """
        if reserved is None:
            reserved = max(1, self.remaining // 2)
        return IterationBudget(
            max_iterations=reserved,
            _parent=self,
        )

    def reset(self) -> None:
        """Reset the counter (e.g. for a new conversation turn)."""
        with self._lock:
            self._used = 0

    def __str__(self) -> str:
        return f"IterationBudget({self.used}/{self.max_iterations})"
