from __future__ import annotations
from homie_core.plugins.base import HomiePlugin, PluginResult
from homie_core.utils import utc_now


class HealthPlugin(HomiePlugin):
    name = "health"
    description = "Break reminders and screen time tracking"
    permissions = ["read_activity"]

    def __init__(self):
        self._session_start = None
        self._total_screen_time: float = 0
        self._break_interval_min: int = 90
        self._last_break = None

    def on_activate(self, config):
        self._break_interval_min = config.get("break_interval", 90)
        self._session_start = utc_now()
        self._last_break = utc_now()

    def on_deactivate(self): pass

    def on_context(self):
        if not self._last_break:
            return {}
        minutes_since_break = (utc_now() - self._last_break).total_seconds() / 60
        return {
            "minutes_since_break": round(minutes_since_break),
            "break_needed": minutes_since_break >= self._break_interval_min,
        }

    def on_query(self, intent, params):
        if intent == "screen_time":
            if self._session_start:
                elapsed = (utc_now() - self._session_start).total_seconds() / 3600
                return PluginResult(success=True, data={"hours": round(elapsed, 1)})
            return PluginResult(success=True, data={"hours": 0})
        if intent == "break_status":
            ctx = self.on_context()
            return PluginResult(success=True, data=ctx)
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        if action == "record_break":
            self._last_break = utc_now()
            return PluginResult(success=True, data="Break recorded")
        return PluginResult(success=False, error=f"Unknown action: {action}")
