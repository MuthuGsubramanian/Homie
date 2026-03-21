# tests/unit/self_healing/test_decorator.py
import time
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.resilience.decorator import resilient
from homie_core.self_healing.resilience.circuit_breaker import CircuitOpenError


class TestResilientDecorator:
    def test_passes_through_on_success(self):
        @resilient(retries=3, base_delay=0.01, timeout=5.0)
        def greet(name):
            return f"hi {name}"

        assert greet("homie") == "hi homie"

    def test_retries_on_transient_error(self):
        call_count = 0

        @resilient(retries=3, base_delay=0.01, timeout=5.0)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timeout")
            return "recovered"

        assert flaky() == "recovered"
        assert call_count == 3

    def test_raises_permanent_error_immediately(self):
        @resilient(retries=3, base_delay=0.01, timeout=5.0)
        def broken():
            raise FileNotFoundError("gone")

        with pytest.raises(FileNotFoundError):
            broken()

    def test_circuit_breaker_trips(self):
        @resilient(
            retries=0,
            base_delay=0.01,
            timeout=5.0,
            circuit_breaker_threshold=2,
            circuit_breaker_window=60,
        )
        def failing():
            raise TimeoutError("down")

        for _ in range(2):
            with pytest.raises(TimeoutError):
                failing()

        with pytest.raises(CircuitOpenError):
            failing()

    def test_timeout_enforcement(self):
        @resilient(retries=0, base_delay=0.01, timeout=0.1)
        def slow():
            time.sleep(10)

        with pytest.raises(TimeoutError):
            slow()

    def test_fallback_called_on_failure(self):
        fallback_fn = MagicMock(return_value="fallback_result")

        @resilient(retries=0, base_delay=0.01, timeout=5.0, fallback=fallback_fn)
        def broken():
            raise TimeoutError("down")

        result = broken()
        assert result == "fallback_result"
        fallback_fn.assert_called_once()

    def test_fallback_called_on_circuit_open(self):
        fallback_fn = MagicMock(return_value="safe")

        @resilient(
            retries=0,
            base_delay=0.01,
            timeout=5.0,
            circuit_breaker_threshold=1,
            circuit_breaker_window=60,
            fallback=fallback_fn,
        )
        def broken():
            raise TimeoutError("down")

        with pytest.raises(TimeoutError):
            broken()  # trips circuit

        result = broken()  # circuit open, fallback used
        assert result == "safe"

    def test_works_as_method_decorator(self):
        class MyService:
            @resilient(retries=1, base_delay=0.01, timeout=5.0)
            def query(self, q):
                return f"result:{q}"

        svc = MyService()
        assert svc.query("test") == "result:test"

    def test_health_status_exposed(self):
        @resilient(retries=1, base_delay=0.01, timeout=5.0)
        def fn():
            return "ok"

        fn()
        status = fn.health_status()
        assert status["state"] == "healthy"
        assert status["total_successes"] == 1
        assert status["total_failures"] == 0
