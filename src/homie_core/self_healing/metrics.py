"""Time-series metric collection with anomaly detection."""

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnomalyAlert:
    """Alert raised when a metric deviates significantly from its baseline."""

    module: str
    metric_name: str
    current_value: float
    mean: float
    std_dev: float
    threshold_multiplier: float


class MetricsCollector:
    """Collects time-series metrics per module and detects anomalies."""

    def __init__(
        self,
        window_size: int = 200,
        anomaly_std_threshold: float = 3.0,
        min_samples_for_anomaly: int = 10,
    ) -> None:
        self._window_size = window_size
        self._anomaly_threshold = anomaly_std_threshold
        self._min_samples = min_samples_for_anomaly
        self._lock = threading.Lock()
        # {module: {metric_name: deque[float]}}
        self._data: dict[str, dict[str, deque[float]]] = {}

    def record(
        self, module: str, metric_name: str, value: float
    ) -> Optional[AnomalyAlert]:
        """Record a metric value. Returns AnomalyAlert if anomaly detected."""
        with self._lock:
            if module not in self._data:
                self._data[module] = {}
            if metric_name not in self._data[module]:
                self._data[module][metric_name] = deque(maxlen=self._window_size)

            series = self._data[module][metric_name]

            # Check for anomaly before adding new value
            alert = None
            if len(series) >= self._min_samples:
                mean = sum(series) / len(series)
                variance = sum((x - mean) ** 2 for x in series) / len(series)
                std_dev = math.sqrt(variance) if variance > 0 else 0.0

                # Anomaly: value deviates beyond threshold * std_dev from mean.
                # For constant series (std_dev=0), flag if deviation exceeds 10% of mean.
                min_abs_deviation = abs(mean * 0.1) if mean != 0 else 1.0
                if (std_dev > 0 and abs(value - mean) > self._anomaly_threshold * std_dev) or (
                    std_dev == 0.0 and abs(value - mean) > min_abs_deviation
                ):
                    alert = AnomalyAlert(
                        module=module,
                        metric_name=metric_name,
                        current_value=value,
                        mean=mean,
                        std_dev=std_dev,
                        threshold_multiplier=self._anomaly_threshold,
                    )

            series.append(value)
            return alert

    def get_latest(self, module: str, metric_name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        with self._lock:
            series = self._data.get(module, {}).get(metric_name)
            if series and len(series) > 0:
                return series[-1]
            return None

    def get_average(self, module: str, metric_name: str) -> Optional[float]:
        """Get the moving average for a metric."""
        with self._lock:
            series = self._data.get(module, {}).get(metric_name)
            if series and len(series) > 0:
                return sum(series) / len(series)
            return None

    def snapshot(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return a snapshot of all metrics with latest/average values."""
        with self._lock:
            result = {}
            for module, metrics in self._data.items():
                result[module] = {}
                for name, series in metrics.items():
                    if len(series) > 0:
                        result[module][name] = {
                            "latest": series[-1],
                            "average": sum(series) / len(series),
                            "count": len(series),
                        }
            return result
