"""Health probe for LAN discovery and network services."""

from .base import BaseProbe, HealthStatus, ProbeResult


class NetworkProbe(BaseProbe):
    """Checks LAN discovery and WebSocket connection health."""

    name = "network"
    interval = 30.0

    def __init__(self, network_manager=None) -> None:
        self._net = network_manager

    def check(self) -> ProbeResult:
        if self._net is None:
            return ProbeResult(
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                error_count=0,
                last_error="Network manager not initialized",
            )

        try:
            is_running = self._net.is_running
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=str(exc),
            )

        if not is_running:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error="Network service not running",
            )

        peer_count = getattr(self._net, "peer_count", 0)
        status = HealthStatus.HEALTHY if peer_count > 0 else HealthStatus.DEGRADED

        return ProbeResult(
            status=status,
            latency_ms=0,
            error_count=0,
            metadata={"peer_count": peer_count},
        )
