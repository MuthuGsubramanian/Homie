"""Timeout enforcement — prevents operations from hanging indefinitely."""

import threading
from typing import Any, Callable, Optional


def run_with_timeout(
    fn: Callable,
    timeout: float,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    operation_name: str = "operation",
) -> Any:
    """Run fn in a daemon thread with a timeout.

    Returns the result if fn completes within timeout.
    Raises TimeoutError if fn doesn't complete in time.
    Propagates any exception raised by fn.
    """
    kwargs = kwargs or {}
    result: list = [None]
    error: list = [None]

    def _run():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as exc:
            error[0] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(
            f"{operation_name} exceeded {timeout}s timeout"
        )

    if error[0] is not None:
        raise error[0]

    return result[0]
