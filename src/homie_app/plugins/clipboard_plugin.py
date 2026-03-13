from __future__ import annotations
from homie_core.plugins.base import HomiePlugin, PluginResult
from homie_core.context.clipboard import ClipboardMonitor


class ClipboardPlugin(HomiePlugin):
    name = "clipboard"
    description = "Clipboard history and search"
    permissions = ["read_clipboard"]

    def __init__(self):
        self._monitor = ClipboardMonitor()

    def on_activate(self, config): pass
    def on_deactivate(self): pass

    def on_context(self):
        self._monitor.check()
        return {}

    def on_query(self, intent, params):
        if intent == "history":
            n = params.get("n", 10)
            return PluginResult(success=True, data=self._monitor.get_history(n))
        if intent == "search":
            query = params.get("query", "")
            return PluginResult(success=True, data=self._monitor.search(query))
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        return PluginResult(success=False, error="No actions supported")
