"""Health probe for context aggregation observers."""

from .base import BaseProbe, HealthStatus, ProbeResult


class ContextProbe(BaseProbe):
    """Checks context aggregator and observer health."""

    name = "context"
    interval = 30.0

    def __init__(self, context_aggregator=None) -> None:
        self._agg = context_aggregator

    def check(self) -> ProbeResult:
        if self._agg is None:
            return ProbeResult(
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                error_count=0,
                last_error="Context aggregator not initialized",
            )

        try:
            snapshot = self._agg.tick()
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=str(exc),
            )

        if not snapshot:
            return ProbeResult(
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error_count=0,
                last_error="Context tick returned empty snapshot",
            )

        return ProbeResult(
            status=HealthStatus.HEALTHY,
            latency_ms=0,
            error_count=0,
            metadata={"keys": list(snapshot.keys())},
        )
