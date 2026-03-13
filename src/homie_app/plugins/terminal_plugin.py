from __future__ import annotations
from pathlib import Path
from homie_core.plugins.base import HomiePlugin, PluginResult


class TerminalPlugin(HomiePlugin):
    name = "terminal"
    description = "Shell history and recent commands"
    permissions = ["read_shell_history"]

    def on_activate(self, config): pass
    def on_deactivate(self): pass

    def on_query(self, intent, params):
        if intent == "history":
            n = params.get("n", 20)
            history = self._read_history(n)
            return PluginResult(success=True, data=history)
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        return PluginResult(success=False, error="No actions supported")

    def _read_history(self, n: int = 20) -> list[str]:
        candidates = [
            Path.home() / ".bash_history",
            Path.home() / ".zsh_history",
            Path.home() / "AppData/Roaming/Microsoft/Windows/PowerShell/PSReadLine/ConsoleHost_history.txt",
        ]
        for p in candidates:
            if p.exists():
                try:
                    lines = p.read_text(encoding="utf-8", errors="ignore").strip().split("\n")
                    return lines[-n:]
                except Exception:
                    pass
        return []
