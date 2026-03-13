from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from homie_core.config import NotificationConfig


@dataclass
class Notification:
    category: str
    title: str
    body: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


class NotificationRouter:
    def __init__(self, config: NotificationConfig):
        self._config = config
        self._dnd = False
        self._pending: list[Notification] = []

    def set_dnd(self, on: bool) -> None:
        self._dnd = on

    def should_deliver(self, n: Notification) -> bool:
        if self._dnd or self._is_in_dnd_schedule():
            return False
        return self._config.categories.get(n.category, False)

    def route(self, n: Notification) -> bool:
        if self.should_deliver(n):
            return True
        self._pending.append(n)
        return False

    def get_pending(self) -> list[Notification]:
        return list(self._pending)

    def flush_pending(self) -> list[Notification]:
        pending = self._pending
        self._pending = []
        return pending

    def _is_in_dnd_schedule(self, current_time: str | None = None) -> bool:
        if not self._config.dnd_schedule_enabled:
            return False
        now = current_time or datetime.now().strftime("%H:%M")
        start = self._config.dnd_schedule_start
        end = self._config.dnd_schedule_end
        if start <= end:
            return start <= now <= end
        else:  # wraps midnight (e.g., 22:00 - 07:00)
            return now >= start or now < end
