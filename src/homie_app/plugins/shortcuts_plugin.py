from __future__ import annotations
from homie_core.plugins.base import HomiePlugin, PluginResult


class ShortcutsPlugin(HomiePlugin):
    name = "shortcuts"
    description = "User-defined trigger-action macros"
    permissions = ["execute_shortcuts"]

    def __init__(self):
        self._shortcuts: dict[str, dict] = {}

    def on_activate(self, config):
        self._shortcuts = config.get("shortcuts", {})

    def on_deactivate(self): pass

    def on_query(self, intent, params):
        if intent == "list":
            return PluginResult(success=True, data=list(self._shortcuts.items()))
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        if action == "add":
            name = params.get("name", "")
            trigger = params.get("trigger", "")
            response = params.get("response", "")
            if name and trigger and response:
                self._shortcuts[name] = {"trigger": trigger, "response": response}
                return PluginResult(success=True, data=f"Shortcut '{name}' added")
            return PluginResult(success=False, error="Missing name, trigger, or response")
        if action == "remove":
            name = params.get("name", "")
            self._shortcuts.pop(name, None)
            return PluginResult(success=True, data=f"Shortcut '{name}' removed")
        return PluginResult(success=False, error=f"Unknown action: {action}")
