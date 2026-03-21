"""Retry logic with exponential backoff and jitter."""

import random
import time
from typing import Any, Callable, Optional

from .exceptions import ErrorCategory, classify_exception


def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    custom_rules: Optional[dict[type, ErrorCategory]] = None,
    on_retry: Optional[Callable] = None,
) -> Any:
    """Call fn with retries on transient/recoverable errors.

    Uses exponential backoff with jitter. Permanent and fatal errors
    are raised immediately without retry.
    """
    kwargs = kwargs or {}
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            category = classify_exception(exc, custom_rules)

            if category in (ErrorCategory.PERMANENT, ErrorCategory.FATAL):
                raise

            if attempt >= max_retries:
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            # Add jitter: ±50% of delay
            delay = delay * (0.5 + random.random())

            if on_retry:
                on_retry(attempt=attempt, exception=exc, category=category, delay=delay)

            time.sleep(delay)

    raise last_exc  # unreachable but satisfies type checker
