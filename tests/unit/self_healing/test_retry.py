# tests/unit/self_healing/test_retry.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.resilience.retry import retry_with_backoff
from homie_core.self_healing.resilience.exceptions import ErrorCategory


class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        fn = MagicMock(return_value="ok")
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 1

    def test_retries_on_transient_error(self):
        fn = MagicMock(side_effect=[TimeoutError("timeout"), "ok"])
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 2

    def test_no_retry_on_permanent_error(self):
        fn = MagicMock(side_effect=FileNotFoundError("gone"))
        with pytest.raises(FileNotFoundError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert fn.call_count == 1

    def test_no_retry_on_fatal_error(self):
        fn = MagicMock(side_effect=MemoryError())
        with pytest.raises(MemoryError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert fn.call_count == 1

    def test_exhausts_retries_then_raises(self):
        fn = MagicMock(side_effect=TimeoutError("timeout"))
        with pytest.raises(TimeoutError):
            retry_with_backoff(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 3  # 1 initial + 2 retries

    def test_passes_args_and_kwargs(self):
        fn = MagicMock(return_value="ok")
        retry_with_backoff(fn, max_retries=1, base_delay=0.01, args=("a",), kwargs={"b": 2})
        fn.assert_called_once_with("a", b=2)

    def test_exponential_backoff_timing(self):
        fn = MagicMock(side_effect=[TimeoutError(), TimeoutError(), "ok"])
        with patch("homie_core.self_healing.resilience.retry.time.sleep") as mock_sleep:
            retry_with_backoff(fn, max_retries=3, base_delay=1.0)
            # First retry: 1.0s, second retry: 2.0s (exponential)
            delays = [call.args[0] for call in mock_sleep.call_args_list]
            assert len(delays) == 2
            assert delays[0] == pytest.approx(1.0, abs=0.5)  # jitter
            assert delays[1] == pytest.approx(2.0, abs=1.0)

    def test_custom_classifier(self):
        custom = {RuntimeError: ErrorCategory.TRANSIENT}
        fn = MagicMock(side_effect=[RuntimeError("temp"), "ok"])
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01, custom_rules=custom)
        assert result == "ok"
        assert fn.call_count == 2

    def test_on_retry_callback(self):
        callback = MagicMock()
        fn = MagicMock(side_effect=[TimeoutError(), "ok"])
        retry_with_backoff(fn, max_retries=3, base_delay=0.01, on_retry=callback)
        assert callback.call_count == 1
        call_args = callback.call_args
        assert call_args[1]["attempt"] == 0
        assert isinstance(call_args[1]["exception"], TimeoutError)
