from __future__ import annotations

from typing import Any, Optional

from homie_core.context.screen_monitor import ScreenMonitor
from homie_core.context.app_tracker import AppTracker
from homie_core.context.clipboard import ClipboardMonitor
from homie_core.memory.working import WorkingMemory
from homie_core.utils import utc_now


class ContextAggregator:
    def __init__(
        self,
        working_memory: WorkingMemory,
        screen_monitor: Optional[ScreenMonitor] = None,
        app_tracker: Optional[AppTracker] = None,
        clipboard_monitor: Optional[ClipboardMonitor] = None,
    ):
        self._wm = working_memory
        self._screen = screen_monitor or ScreenMonitor()
        self._apps = app_tracker or AppTracker()
        self._clipboard = clipboard_monitor or ClipboardMonitor()

    def tick(self) -> dict[str, Any]:
        snapshot = {}

        # Screen
        window = self._screen.get_active_window()
        snapshot["active_window"] = window.title
        snapshot["active_process"] = window.process_name
        self._apps.track(window.process_name)

        # App stats
        snapshot["is_deep_work"] = self._apps.is_deep_work()
        snapshot["switch_count_30m"] = self._apps.get_switch_count(minutes=30)

        # Clipboard
        new_clip = self._clipboard.check()
        if new_clip:
            snapshot["last_clipboard"] = new_clip[:200]

        # Push to working memory
        for k, v in snapshot.items():
            self._wm.update(k, v)

        snapshot["timestamp"] = utc_now().isoformat()
        return snapshot

    def get_full_context(self) -> dict[str, Any]:
        return {
            "working_memory": self._wm.snapshot(),
            "top_apps": self._apps.get_top_apps(5),
            "is_deep_work": self._apps.is_deep_work(),
            "clipboard_history_count": len(self._clipboard.get_history()),
        }
