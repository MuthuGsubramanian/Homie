"""Tests for homie_core.common.retry."""

import pytest

from homie_core.common.retry import retry


class TestRetrySuccess:
    def test_no_retry_needed(self):
        call_count = 0

        @retry(max_attempts=3, delay=0)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_succeeds_after_transient_failure(self):
        attempts = 0

        @retry(max_attempts=3, delay=0)
        def flaky():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("not yet")
            return "done"

        assert flaky() == "done"
        assert attempts == 3


class TestRetryExhaustion:
    def test_raises_after_max_attempts(self):
        attempts = 0

        @retry(max_attempts=2, delay=0)
        def always_fail():
            nonlocal attempts
            attempts += 1
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            always_fail()
        assert attempts == 2


class TestRetrySpecificExceptions:
    def test_retries_only_listed_exceptions(self):
        attempts = 0

        @retry(max_attempts=3, delay=0, exceptions=(ValueError,))
        def wrong_error():
            nonlocal attempts
            attempts += 1
            raise TypeError("nope")

        with pytest.raises(TypeError):
            wrong_error()
        # Should NOT have retried — TypeError is not in the tuple.
        assert attempts == 1

    def test_retries_matching_exception(self):
        attempts = 0

        @retry(max_attempts=3, delay=0, exceptions=(ValueError,))
        def right_error():
            nonlocal attempts
            attempts += 1
            raise ValueError("yes")

        with pytest.raises(ValueError):
            right_error()
        assert attempts == 3


class TestRetryCallback:
    def test_on_retry_called(self):
        log = []

        @retry(max_attempts=3, delay=0, on_retry=lambda a, e: log.append(a))
        def flaky():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            flaky()
        # on_retry is called before retries 2 and 3 (attempts 1 and 2).
        assert log == [1, 2]


class TestRetryValidation:
    def test_max_attempts_zero_raises(self):
        with pytest.raises(ValueError, match="max_attempts"):
            retry(max_attempts=0)
