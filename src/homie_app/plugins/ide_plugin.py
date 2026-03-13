from __future__ import annotations
import json
from pathlib import Path
from homie_core.plugins.base import HomiePlugin, PluginResult


class IDEPlugin(HomiePlugin):
    name = "ide"
    description = "IDE integration (VS Code, JetBrains)"
    permissions = ["read_ide_state"]

    def on_activate(self, config): pass
    def on_deactivate(self): pass

    def on_query(self, intent, params):
        if intent == "recent_projects":
            projects = self._get_vscode_recent()
            return PluginResult(success=True, data=projects)
        if intent == "extensions":
            exts = self._get_vscode_extensions()
            return PluginResult(success=True, data=exts)
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        return PluginResult(success=False, error="No actions supported")

    def _get_vscode_recent(self) -> list[str]:
        storage_path = Path.home() / "AppData/Roaming/Code/User/globalStorage/storage.json"
        if not storage_path.exists():
            storage_path = Path.home() / "AppData/Roaming/Code/storage.json"
        try:
            if storage_path.exists():
                data = json.loads(storage_path.read_text(encoding="utf-8", errors="ignore"))
                entries = data.get("openedPathsList", {}).get("entries", [])
                return [str(e.get("folderUri", e.get("fileUri", ""))) for e in entries[:10]]
        except Exception:
            pass
        return []

    def _get_vscode_extensions(self) -> list[str]:
        ext_dir = Path.home() / ".vscode/extensions"
        if ext_dir.exists():
            return [d.name for d in ext_dir.iterdir() if d.is_dir()][:20]
        return []
