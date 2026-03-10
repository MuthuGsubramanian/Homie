from __future__ import annotations

import threading
from typing import Callable, Optional

try:
    from pynput import keyboard
except ImportError:
    keyboard = None  # type: ignore[assignment]


_HOTKEY_MAP = {
    "alt+8": "<alt>+8",
    "alt+h": "<alt>+h",
    "ctrl+space": "<ctrl>+<space>",
    "f9": "<f9>",
}


class HotkeyListener:
    """Registers a global hotkey and calls a callback when pressed."""

    def __init__(self, hotkey: str = "alt+8", callback: Optional[Callable] = None):
        self._hotkey = hotkey
        self._callback = callback
        self._running = False
        self._listener = None
        self._thread: Optional[threading.Thread] = None

    def _on_activate(self) -> None:
        if self._callback:
            self._callback()

    def start(self) -> None:
        if keyboard is None:
            return
        pynput_hotkey = _HOTKEY_MAP.get(self._hotkey, self._hotkey)
        self._listener = keyboard.GlobalHotKeys({
            pynput_hotkey: self._on_activate,
        })
        self._listener.start()
        self._running = True

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running
