from __future__ import annotations
from pathlib import Path
from homie_core.plugins.base import HomiePlugin, PluginResult


class NotesPlugin(HomiePlugin):
    name = "notes"
    description = "Search and read local markdown/text notes"
    permissions = ["read_files"]

    def __init__(self):
        self._notes_dirs: list[Path] = []

    def on_activate(self, config):
        dirs = config.get("directories", [])
        self._notes_dirs = [Path(d) for d in dirs if Path(d).exists()]
        if not self._notes_dirs:
            # Auto-detect common locations
            for candidate in [Path.home() / "Documents", Path.home() / "Notes", Path.home() / "Obsidian"]:
                if candidate.exists():
                    self._notes_dirs.append(candidate)

    def on_deactivate(self): pass

    def on_query(self, intent, params):
        if intent == "search":
            query = params.get("query", "")
            results = self._search_notes(query)
            return PluginResult(success=True, data=results)
        if intent == "recent":
            n = params.get("n", 10)
            results = self._recent_notes(n)
            return PluginResult(success=True, data=results)
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        return PluginResult(success=False, error="No actions supported")

    def _search_notes(self, query: str) -> list[dict]:
        results = []
        query_lower = query.lower()
        for d in self._notes_dirs:
            for f in d.rglob("*.md"):
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    if query_lower in content.lower() or query_lower in f.name.lower():
                        results.append({"path": str(f), "name": f.name, "preview": content[:200]})
                except Exception:
                    pass
        return results[:20]

    def _recent_notes(self, n: int = 10) -> list[dict]:
        all_notes = []
        for d in self._notes_dirs:
            for f in d.rglob("*.md"):
                try:
                    all_notes.append({"path": str(f), "name": f.name, "modified": f.stat().st_mtime})
                except Exception:
                    pass
        all_notes.sort(key=lambda x: x["modified"], reverse=True)
        return all_notes[:n]
