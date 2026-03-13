from __future__ import annotations

import threading
from collections import deque
from typing import Any

from homie_core.utils import utc_now


class WorkingMemory:
    def __init__(self, max_age_seconds: int = 300, max_conversation_turns: int = 50):
        self._state: dict[str, Any] = {}
        self._conversation: deque[dict] = deque(maxlen=max_conversation_turns)
        self._max_age = max_age_seconds
        self._lock = threading.Lock()

    def update(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def add_message(self, role: str, content: str) -> None:
        with self._lock:
            self._conversation.append({
                "role": role,
                "content": content,
                "timestamp": utc_now().isoformat(),
            })

    def get_conversation(self) -> list[dict]:
        with self._lock:
            return list(self._conversation)

    def clear(self) -> None:
        with self._lock:
            self._state.clear()
            self._conversation.clear()
