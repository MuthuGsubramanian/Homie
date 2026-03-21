"""Resource scheduler — learns usage patterns for proactive resource management."""

import threading
from collections import defaultdict


class ResourceScheduler:
    """Learns hourly activity patterns to predict resource needs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # {hour: {activity: count}}
        self._hourly: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_activity(self, hour: int, activity: str) -> None:
        """Record an activity observation at a given hour."""
        with self._lock:
            self._hourly[hour][activity] += 1

    def get_hour_pattern(self, hour: int) -> dict[str, int]:
        """Get the activity frequency pattern for an hour."""
        with self._lock:
            return dict(self._hourly.get(hour, {}))

    def predict_activity(self, hour: int) -> str:
        """Predict the most likely activity for a given hour."""
        with self._lock:
            pattern = self._hourly.get(hour)
            if not pattern:
                return "idle"
            return max(pattern, key=pattern.get)

    def should_preload(self, hour: int) -> bool:
        """Should the model be pre-loaded for this hour?"""
        prediction = self.predict_activity(hour)
        return prediction in ("inference", "coding", "conversation")

    def get_schedule_summary(self) -> dict[int, str]:
        """Get a 24-hour schedule summary of predicted activities."""
        summary = {}
        for hour in sorted(self._hourly.keys()):
            summary[hour] = self.predict_activity(hour)
        return summary
