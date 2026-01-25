from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass
class TrayActions:
    start_services: Callable[[], None]
    stop_services: Callable[[], None]
    toggle_recording: Callable[[], None]
    pause_suggestions: Callable[[bool], None]
    clear_data: Callable[[], None]


def launch_tray(actions: TrayActions) -> None:
    """Placeholder tray launcher.

    Real implementation should use a lightweight library (e.g., pystray) with visible
    menu items. Here we expose a minimal CLI fallback to avoid heavy deps.
    """
    print("HOMIE tray fallback running. Commands: start/stop/record/pause/resume/clear/exit")
    while True:
        cmd = input("tray> ").strip().lower()
        if cmd == "start":
            actions.start_services()
        elif cmd == "stop":
            actions.stop_services()
        elif cmd == "record":
            actions.toggle_recording()
        elif cmd == "pause":
            actions.pause_suggestions(True)
        elif cmd == "resume":
            actions.pause_suggestions(False)
        elif cmd == "clear":
            actions.clear_data()
        elif cmd == "dashboard":
            subprocess.Popen([sys.executable, "-m", "webbrowser", "http://localhost:8080"])
        elif cmd in {"exit", "quit"}:
            break
        else:
            print("unknown command")


__all__ = ["TrayActions", "launch_tray"]
