from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal


@dataclass
class CliNotification:
    text: str
    category: str
    priority: Literal["urgent", "normal", "low"] = "normal"


class NotificationQueue:
    def __init__(self, max_size: int = 10):
        self._queue: deque[CliNotification] = deque(maxlen=max_size)

    def push(self, notification: CliNotification) -> None:
        self._queue.append(notification)

    def drain(self) -> list[CliNotification]:
        items = list(self._queue)
        self._queue.clear()
        return items

    def has_urgent(self) -> bool:
        return any(n.priority == "urgent" for n in self._queue)

    def __len__(self) -> int:
        return len(self._queue)
