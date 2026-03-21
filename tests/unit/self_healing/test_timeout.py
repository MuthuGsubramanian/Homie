# tests/unit/self_healing/test_timeout.py
import time
import pytest
from homie_core.self_healing.resilience.timeout import run_with_timeout


class TestRunWithTimeout:
    def test_returns_result_within_timeout(self):
        result = run_with_timeout(lambda: "ok", timeout=5.0)
        assert result == "ok"

    def test_raises_timeout_on_slow_function(self):
        def slow():
            time.sleep(10)
        with pytest.raises(TimeoutError, match="exceeded 0.1s"):
            run_with_timeout(slow, timeout=0.1)

    def test_propagates_function_exception(self):
        def bad():
            raise ValueError("broken")
        with pytest.raises(ValueError, match="broken"):
            run_with_timeout(bad, timeout=5.0)

    def test_passes_args_and_kwargs(self):
        def add(a, b=0):
            return a + b
        result = run_with_timeout(add, timeout=5.0, args=(3,), kwargs={"b": 4})
        assert result == 7

    def test_custom_timeout_message(self):
        def slow():
            time.sleep(10)
        with pytest.raises(TimeoutError, match="model inference"):
            run_with_timeout(slow, timeout=0.1, operation_name="model inference")
