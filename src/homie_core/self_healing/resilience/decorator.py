"""The @resilient decorator — wraps functions with retry, circuit breaker, and timeout."""

import functools
from typing import Any, Callable, Optional

from .circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from .exceptions import ErrorCategory, classify_exception
from .retry import retry_with_backoff
from .timeout import run_with_timeout


def resilient(
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    timeout: float = 30.0,
    circuit_breaker_threshold: int = 5,
    circuit_breaker_window: float = 60.0,
    circuit_breaker_cooldown: float = 30.0,
    fallback: Optional[Callable] = None,
    custom_rules: Optional[dict[type, ErrorCategory]] = None,
) -> Callable:
    """Decorator that adds retry, circuit breaker, and timeout to a function."""

    def decorator(fn: Callable) -> Callable:
        cb = CircuitBreaker(
            threshold=circuit_breaker_threshold,
            window=circuit_breaker_window,
            cooldown=circuit_breaker_cooldown,
            name=getattr(fn, "__qualname__", fn.__name__),
        )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check circuit breaker first
            try:
                cb.check()
            except CircuitOpenError:
                if fallback is not None:
                    return fallback(*args, **kwargs)
                raise

            def _call_with_timeout():
                return run_with_timeout(
                    fn,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs,
                    operation_name=fn.__qualname__ if hasattr(fn, "__qualname__") else fn.__name__,
                )

            def _on_retry(attempt, exception, category, delay):
                cb.record_failure()

            try:
                result = retry_with_backoff(
                    _call_with_timeout,
                    max_retries=retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    custom_rules=custom_rules,
                    on_retry=_on_retry,
                )
                cb.record_success()
                return result
            except Exception as exc:
                cb.record_failure()
                # Use fallback only if the circuit hasn't just tripped open
                if fallback is not None and cb.state != CircuitState.OPEN:
                    return fallback(*args, **kwargs)
                raise

        def health_status() -> dict[str, Any]:
            """Return current health metrics for this function."""
            state = cb.state
            return {
                "state": "healthy" if state.value == "closed" else "degraded" if state.value == "half_open" else "failed",
                "circuit_state": state.value,
                "total_successes": cb.total_successes,
                "total_failures": cb.total_failures,
                "trip_count": cb.trip_count,
            }

        wrapper.health_status = health_status
        wrapper._circuit_breaker = cb
        return wrapper

    return decorator
