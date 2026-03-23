"""Detects patterns in time-series and categorical data — periodicity, anomalies, and trends."""

from __future__ import annotations

import math
from typing import Any


class PatternDetector:
    """Pure-Python pattern recognition for time-series and categorical data.

    No external dependencies (numpy/scipy) required.
    """

    # ------------------------------------------------------------------
    # Periodicity detection
    # ------------------------------------------------------------------

    def detect_periodicity(
        self,
        timestamps: list[float],
        values: list[Any],
    ) -> dict:
        """Detect periodic patterns in the time-series.

        Returns a dict with keys:
        - ``is_periodic`` — bool
        - ``period`` — estimated period (in same units as timestamps) or ``None``
        - ``confidence`` — 0..1 score
        - ``details`` — extra information
        """
        if len(timestamps) < 4:
            return {"is_periodic": False, "period": None, "confidence": 0.0, "details": "too few data points"}

        # Compute intervals between successive events with the same value
        value_times: dict[Any, list[float]] = {}
        for ts, val in zip(timestamps, values):
            value_times.setdefault(val, []).append(ts)

        best_period: float | None = None
        best_confidence = 0.0

        for val, times in value_times.items():
            if len(times) < 3:
                continue
            times_sorted = sorted(times)
            gaps = [times_sorted[i + 1] - times_sorted[i] for i in range(len(times_sorted) - 1)]
            if not gaps:
                continue

            mean_gap = sum(gaps) / len(gaps)
            if mean_gap == 0:
                continue

            variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
            std = math.sqrt(variance)
            cv = std / mean_gap  # coefficient of variation — lower is more periodic

            confidence = max(0.0, 1.0 - cv)
            if confidence > best_confidence:
                best_confidence = confidence
                best_period = mean_gap

        is_periodic = best_confidence >= 0.5
        return {
            "is_periodic": is_periodic,
            "period": best_period if is_periodic else None,
            "confidence": round(best_confidence, 4),
            "details": f"best_period={best_period}, n_values={len(value_times)}",
        }

    # ------------------------------------------------------------------
    # Anomaly detection (z-score based)
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        values: list[float],
        threshold: float = 2.0,
    ) -> list[int]:
        """Return indices of values that are anomalous (|z-score| > *threshold*).

        Uses mean/stddev computed over the full series.
        """
        if len(values) < 2:
            return []

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        if std == 0:
            return []

        return [i for i, v in enumerate(values) if abs((v - mean) / std) > threshold]

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def detect_trends(self, values: list[float]) -> str:
        """Return ``"increasing"``, ``"decreasing"``, or ``"stable"``.

        Uses simple linear regression (ordinary least squares) over the
        index positions.
        """
        n = len(values)
        if n < 2:
            return "stable"

        # Compute slope via OLS:  slope = Cov(x, y) / Var(x)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Normalise slope relative to the data range for a stable threshold
        data_range = max(values) - min(values)
        if data_range == 0:
            return "stable"

        normalised = slope * n / data_range  # dimensionless slope metric

        if normalised > 0.3:
            return "increasing"
        elif normalised < -0.3:
            return "decreasing"
        return "stable"

    # ------------------------------------------------------------------
    # Frequency analysis (bonus helper)
    # ------------------------------------------------------------------

    def value_frequencies(self, values: list[Any]) -> dict[Any, int]:
        """Return a frequency count of each distinct value."""
        freq: dict[Any, int] = {}
        for v in values:
            freq[v] = freq.get(v, 0) + 1
        return freq
