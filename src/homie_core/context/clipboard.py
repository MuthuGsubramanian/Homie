from __future__ import annotations

import platform
import threading
from collections import deque
from typing import Optional

from homie_core.utils import utc_now


class ClipboardMonitor:
    def __init__(self, max_history: int = 50):
        self._history: deque[dict] = deque(maxlen=max_history)
        self._last_content: Optional[str] = None
        self._lock = threading.Lock()

    def check(self) -> Optional[str]:
        content = self._get_clipboard()
        if content and content != self._last_content:
            with self._lock:
                self._last_content = content
                self._history.append({
                    "content": content[:500],  # truncate long clips
                    "timestamp": utc_now().isoformat(),
                })
            return content
        return None

    def get_history(self, n: int = 10) -> list[dict]:
        with self._lock:
            return list(self._history)[-n:]

    def search(self, query: str) -> list[dict]:
        query_lower = query.lower()
        with self._lock:
            return [h for h in self._history if query_lower in h["content"].lower()]

    def _get_clipboard(self) -> Optional[str]:
        if platform.system() == "Windows":
            try:
                import win32clipboard
                win32clipboard.OpenClipboard()
                try:
                    data = win32clipboard.GetClipboardData()
                    return str(data)
                except (TypeError, win32clipboard.error):
                    return None
                finally:
                    win32clipboard.CloseClipboard()
            except ImportError:
                return None
            except Exception:
                return None
        return None
