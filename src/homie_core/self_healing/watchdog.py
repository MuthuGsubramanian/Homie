"""HealthWatchdog — central coordinator for self-healing runtime."""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from .event_bus import EventBus, HealthEvent
from .health_log import HealthLog
from .metrics import MetricsCollector
from .probes.base import BaseProbe, HealthStatus, ProbeResult

logger = logging.getLogger(__name__)


class HealthWatchdog:
    """Central health monitoring and recovery coordination service."""

    def __init__(
        self,
        db_path: Path | str,
        probe_interval: float = 30.0,
    ) -> None:
        self._db_path = Path(db_path)
        self._probe_interval = probe_interval
        self._probes: dict[str, BaseProbe] = {}
        self._last_results: dict[str, ProbeResult] = {}
        self._recovery_engine = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Initialize subsystems
        self.event_bus = EventBus()
        self.health_log = HealthLog(db_path=self._db_path)
        self.health_log.initialize()
        self.metrics = MetricsCollector()

    @property
    def registered_probes(self) -> dict[str, BaseProbe]:
        return dict(self._probes)

    @property
    def system_health(self) -> HealthStatus:
        """Overall system health — worst probe status wins."""
        if not self._last_results:
            return HealthStatus.UNKNOWN
        worst = HealthStatus.HEALTHY
        for result in self._last_results.values():
            if result.status > worst:
                worst = result.status
        return worst

    def register_probe(self, probe: BaseProbe) -> None:
        """Register a health probe."""
        self._probes[probe.name] = probe

    def set_recovery_engine(self, recovery_engine) -> None:
        """Set the recovery engine for handling failures."""
        self._recovery_engine = recovery_engine

    def run_all_probes(self) -> dict[str, ProbeResult]:
        """Run all registered probes and return results."""
        results = {}
        for name, probe in self._probes.items():
            result = probe.run()
            results[name] = result
            self._last_results[name] = result

            # Record metrics
            self.metrics.record(name, "latency_ms", result.latency_ms)
            self.metrics.record(name, "error_count", float(result.error_count))

            # Log event
            event = HealthEvent(
                module=name,
                event_type="probe_result",
                severity="info" if result.status == HealthStatus.HEALTHY else "warning" if result.status == HealthStatus.DEGRADED else "error",
                details=result.to_dict(),
            )
            self.event_bus.publish(event)
            self.health_log.write(event)

            # Trigger recovery if needed
            if result.status >= HealthStatus.FAILED and self._recovery_engine:
                self._recovery_engine.recover(
                    module=name,
                    status=result.status,
                    error=result.last_error or "probe failed",
                )

        return results

    def start(self) -> None:
        """Start the watchdog monitoring loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("HealthWatchdog started (interval: %.1fs)", self._probe_interval)

    def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self.event_bus.shutdown()
        self.health_log.close()
        logger.info("HealthWatchdog stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self.run_all_probes()
            except Exception:
                logger.exception("Watchdog probe cycle failed")

            # Sleep in small increments for responsive shutdown
            elapsed = 0.0
            while elapsed < self._probe_interval and self._running:
                time.sleep(0.1)
                elapsed += 0.1
