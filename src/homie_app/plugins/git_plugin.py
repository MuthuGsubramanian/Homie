from __future__ import annotations
import subprocess
from homie_core.plugins.base import HomiePlugin, PluginResult


class GitPlugin(HomiePlugin):
    name = "git"
    description = "Git repository status and history"
    permissions = ["read_git"]

    def on_activate(self, config):
        self._repo_path = config.get("repo_path", ".")

    def on_deactivate(self): pass

    def on_query(self, intent, params):
        if intent == "status":
            return PluginResult(success=True, data=self._run_git("status", "--porcelain"))
        if intent == "log":
            n = params.get("n", 10)
            return PluginResult(success=True, data=self._run_git("log", "--oneline", f"-{n}"))
        if intent == "branch":
            return PluginResult(success=True, data=self._run_git("branch", "--show-current"))
        if intent == "diff":
            return PluginResult(success=True, data=self._run_git("diff", "--stat"))
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        return PluginResult(success=False, error="Git actions not supported for safety")

    def _run_git(self, *args) -> str:
        try:
            result = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10, cwd=self._repo_path
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""
