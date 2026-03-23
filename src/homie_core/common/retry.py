"""Generic retry decorator for production use.

Retries a function call a configurable number of times with a fixed delay
between attempts, optionally narrowed to specific exception types.
"""

from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[[F], F]:
    """Retry a function up to *max_attempts* times.

    Parameters
    ----------
    max_attempts:
        Total number of attempts (including the first call).
    delay:
        Seconds to sleep between attempts.
    exceptions:
        Exception types that trigger a retry.  Exceptions not in this
        tuple propagate immediately.
    on_retry:
        Optional callback ``(attempt, exception)`` invoked before each
        retry sleep — useful for logging.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        if on_retry is not None:
                            on_retry(attempt, exc)
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
