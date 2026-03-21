# tests/unit/self_healing/test_circuit_breaker.py
import time
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
)


class TestCircuitState:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(threshold=3, window=60, cooldown=5)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3, window=60, cooldown=5)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(threshold=3, window=60, cooldown=5)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(threshold=3, window=60, cooldown=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # count reset to 1

    def test_old_failures_outside_window_dont_count(self):
        cb = CircuitBreaker(threshold=3, window=0.1, cooldown=5)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # old failures expired


class TestCircuitOpenBehavior:
    def test_raises_circuit_open_error(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=5)
        cb.record_failure()
        with pytest.raises(CircuitOpenError):
            cb.check()

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=0.1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_one_call(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=0.1)
        cb.record_failure()
        time.sleep(0.15)
        cb.check()  # should not raise — allows test call

    def test_success_in_half_open_closes_circuit(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=0.1)
        cb.record_failure()
        time.sleep(0.15)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=0.1)
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitMetrics:
    def test_tracks_total_failures(self):
        cb = CircuitBreaker(threshold=5, window=60, cooldown=5)
        for _ in range(3):
            cb.record_failure()
        assert cb.total_failures == 3

    def test_tracks_total_successes(self):
        cb = CircuitBreaker(threshold=5, window=60, cooldown=5)
        for _ in range(3):
            cb.record_success()
        assert cb.total_successes == 3

    def test_tracks_trip_count(self):
        cb = CircuitBreaker(threshold=1, window=60, cooldown=0.05)
        cb.record_failure()  # trip 1
        time.sleep(0.06)
        cb.record_failure()  # trip 2
        assert cb.trip_count == 2
