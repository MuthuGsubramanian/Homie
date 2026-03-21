# tests/unit/self_healing/test_exceptions.py
import sqlite3
import pytest
from homie_core.self_healing.resilience.exceptions import (
    classify_exception,
    ErrorCategory,
)


class TestErrorCategory:
    def test_enum_values(self):
        assert ErrorCategory.TRANSIENT.value == "transient"
        assert ErrorCategory.RECOVERABLE.value == "recoverable"
        assert ErrorCategory.PERMANENT.value == "permanent"
        assert ErrorCategory.FATAL.value == "fatal"


class TestClassifyException:
    def test_timeout_is_transient(self):
        assert classify_exception(TimeoutError("timed out")) == ErrorCategory.TRANSIENT

    def test_connection_error_is_transient(self):
        assert classify_exception(ConnectionError("reset")) == ErrorCategory.TRANSIENT

    def test_oserror_errno_enomem_is_transient(self):
        err = OSError(12, "Cannot allocate memory")
        assert classify_exception(err) == ErrorCategory.TRANSIENT

    def test_sqlite_locked_is_recoverable(self):
        err = sqlite3.OperationalError("database is locked")
        assert classify_exception(err) == ErrorCategory.RECOVERABLE

    def test_file_not_found_is_permanent(self):
        assert classify_exception(FileNotFoundError("no such file")) == ErrorCategory.PERMANENT

    def test_value_error_is_permanent(self):
        assert classify_exception(ValueError("bad config")) == ErrorCategory.PERMANENT

    def test_disk_full_is_fatal(self):
        err = OSError(28, "No space left on device")
        assert classify_exception(err) == ErrorCategory.FATAL

    def test_unknown_exception_defaults_to_permanent(self):
        assert classify_exception(Exception("unknown")) == ErrorCategory.PERMANENT

    def test_custom_classifier_overrides_default(self):
        custom = {RuntimeError: ErrorCategory.TRANSIENT}
        result = classify_exception(RuntimeError("custom"), custom_rules=custom)
        assert result == ErrorCategory.TRANSIENT

    def test_memory_error_is_fatal(self):
        assert classify_exception(MemoryError()) == ErrorCategory.FATAL
