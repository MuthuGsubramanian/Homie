"""Circuit breaker — prevents repeated calls to failing operations."""

import threading
import time
from enum import Enum
from typing import Optional


class CircuitState(str, Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing — reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and rejecting calls."""

    def __init__(self, name: str = "unknown"):
        super().__init__(f"Circuit breaker '{name}' is open — operation rejected")
        self.circuit_name = name


class CircuitBreaker:
    """Tracks failures and trips open to prevent cascading failures."""

    def __init__(
        self,
        threshold: int = 5,
        window: float = 60.0,
        cooldown: float = 30.0,
        name: str = "default",
    ):
        self._threshold = threshold
        self._window = window
        self._cooldown = cooldown
        self._name = name
        self._lock = threading.Lock()

        self._failure_times: list[float] = []
        self._state = CircuitState.CLOSED
        self._opened_at: Optional[float] = None

        # Metrics
        self.total_failures: int = 0
        self.total_successes: int = 0
        self.trip_count: int = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._check_half_open()
            return self._state

    def check(self) -> None:
        """Check if the circuit allows a call. Raises CircuitOpenError if not."""
        state = self.state
        if state == CircuitState.OPEN:
            raise CircuitOpenError(self._name)
        # CLOSED and HALF_OPEN allow the call through

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._check_half_open()
            now = time.monotonic()
            self.total_failures += 1
            self._failure_times.append(now)

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = now
                self.trip_count += 1
                return

            # Prune failures outside window
            cutoff = now - self._window
            self._failure_times = [t for t in self._failure_times if t > cutoff]

            if len(self._failure_times) >= self._threshold:
                self._state = CircuitState.OPEN
                self._opened_at = now
                self.trip_count += 1

    def _check_half_open(self) -> None:
        """Transition OPEN -> HALF_OPEN if cooldown elapsed. Must hold _lock."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self._cooldown:
                self._state = CircuitState.HALF_OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._check_half_open()
            self.total_successes += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_times.clear()
            elif self._state == CircuitState.CLOSED:
                self._failure_times.clear()
