from __future__ import annotations

import logging
import platform
import subprocess
from typing import Optional


def notify(title: str, message: str, data_used: Optional[str] = None) -> None:
    """Lightweight local notification."""
    system = platform.system().lower()
    suffix = f"\n(data: {data_used})" if data_used else ""
    if system == "windows":
        try:
            # Using built-in PowerShell toast (no extra deps); best-effort.
            script = (
                f"[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
                f"$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
                f"$template.GetElementsByTagName('text')[0].AppendChild($template.CreateTextNode('{title}')) > $null;"
                f"$template.GetElementsByTagName('text')[1].AppendChild($template.CreateTextNode('{message}{suffix}')) > $null;"
                f"$toast = [Windows.UI.Notifications.ToastNotification]::new($template);"
                f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('HOMIE').Show($toast);"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", script], check=False)
            return
        except Exception:  # noqa: BLE001
            logging.debug("Windows toast failed, falling back to log")
    elif system == "linux":
        try:
            subprocess.run(["notify-send", title, f"{message}{suffix}"], check=False)
            return
        except FileNotFoundError:
            logging.debug("notify-send missing, fallback to log")

    logging.info("NOTIFY %s - %s%s", title, message, suffix)


__all__ = ["notify"]
