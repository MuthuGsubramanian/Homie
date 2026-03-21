"""Exception classifier — categorizes errors for recovery decisions."""

import errno
import sqlite3
from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    TRANSIENT = "transient"      # Retry immediately
    RECOVERABLE = "recoverable"  # Retry with longer backoff
    PERMANENT = "permanent"      # Fail, report to watchdog
    FATAL = "fatal"              # Trip circuit, escalate


# Errno values that indicate fatal disk/resource exhaustion
_FATAL_ERRNOS = {errno.ENOSPC, errno.EDQUOT} if hasattr(errno, "EDQUOT") else {errno.ENOSPC}

# Errno values that indicate transient resource pressure
_TRANSIENT_ERRNOS = {errno.ENOMEM, errno.EAGAIN, errno.EBUSY}

_DEFAULT_RULES: dict[type, ErrorCategory] = {
    TimeoutError: ErrorCategory.TRANSIENT,
    ConnectionError: ErrorCategory.TRANSIENT,
    ConnectionResetError: ErrorCategory.TRANSIENT,
    ConnectionRefusedError: ErrorCategory.TRANSIENT,
    ConnectionAbortedError: ErrorCategory.TRANSIENT,
    BrokenPipeError: ErrorCategory.TRANSIENT,
    InterruptedError: ErrorCategory.TRANSIENT,
    FileNotFoundError: ErrorCategory.PERMANENT,
    PermissionError: ErrorCategory.PERMANENT,
    ValueError: ErrorCategory.PERMANENT,
    TypeError: ErrorCategory.PERMANENT,
    KeyError: ErrorCategory.PERMANENT,
    AttributeError: ErrorCategory.PERMANENT,
    ImportError: ErrorCategory.PERMANENT,
    MemoryError: ErrorCategory.FATAL,
    SystemError: ErrorCategory.FATAL,
}


def classify_exception(
    exc: BaseException,
    custom_rules: Optional[dict[type, ErrorCategory]] = None,
) -> ErrorCategory:
    """Classify an exception into a recovery category."""
    # Custom rules take precedence
    if custom_rules:
        for exc_type, category in custom_rules.items():
            if isinstance(exc, exc_type):
                return category

    # Check default type rules
    for exc_type, category in _DEFAULT_RULES.items():
        if isinstance(exc, exc_type):
            return category

    # SQLite-specific classification
    if isinstance(exc, sqlite3.OperationalError):
        msg = str(exc).lower()
        if "locked" in msg or "busy" in msg:
            return ErrorCategory.RECOVERABLE
        return ErrorCategory.PERMANENT

    # OSError errno-based classification
    if isinstance(exc, OSError) and exc.errno is not None:
        if exc.errno in _FATAL_ERRNOS:
            return ErrorCategory.FATAL
        if exc.errno in _TRANSIENT_ERRNOS:
            return ErrorCategory.TRANSIENT

    return ErrorCategory.PERMANENT
