"""Milestone tracker — determines when to rebuild the Homie model."""

import threading


class MilestoneTracker:
    """Tracks changes since last model push, triggers rebuild at thresholds."""

    def __init__(
        self,
        min_facts: int = 50,
        min_prefs: int = 10,
        min_customs: int = 3,
    ) -> None:
        self._min_facts = min_facts
        self._min_prefs = min_prefs
        self._min_customs = min_customs
        self._lock = threading.Lock()
        self._new_facts = 0
        self._pref_changes = 0
        self._new_customs = 0
        self._manual = False

    def record_new_fact(self) -> None:
        with self._lock:
            self._new_facts += 1

    def record_preference_change(self) -> None:
        with self._lock:
            self._pref_changes += 1

    def record_new_customization(self) -> None:
        with self._lock:
            self._new_customs += 1

    def trigger_manual(self) -> None:
        with self._lock:
            self._manual = True

    def should_rebuild(self) -> bool:
        """Check if any milestone threshold has been crossed."""
        with self._lock:
            if self._manual:
                return True
            return (
                self._new_facts >= self._min_facts
                or self._pref_changes >= self._min_prefs
                or self._new_customs >= self._min_customs
            )

    def reset(self) -> None:
        """Reset counters after a successful rebuild."""
        with self._lock:
            self._new_facts = 0
            self._pref_changes = 0
            self._new_customs = 0
            self._manual = False

    def get_summary(self) -> dict:
        with self._lock:
            return {
                "new_facts": self._new_facts,
                "preference_changes": self._pref_changes,
                "new_customizations": self._new_customs,
                "manual_triggered": self._manual,
                "thresholds": {
                    "min_facts": self._min_facts,
                    "min_prefs": self._min_prefs,
                    "min_customs": self._min_customs,
                },
            }
