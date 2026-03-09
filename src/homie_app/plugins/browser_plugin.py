from __future__ import annotations
import sqlite3
from pathlib import Path
from homie_core.plugins.base import HomiePlugin, PluginResult


class BrowserPlugin(HomiePlugin):
    name = "browser"
    description = "Browser history and bookmarks"
    permissions = ["read_browser_history"]

    def on_activate(self, config):
        self._history_paths = self._find_history_dbs()

    def on_deactivate(self): pass

    def on_query(self, intent, params):
        if intent == "recent_history":
            limit = params.get("limit", 20)
            history = self._read_history(limit)
            return PluginResult(success=True, data=history)
        if intent == "search_history":
            query = params.get("query", "")
            history = self._search_history(query)
            return PluginResult(success=True, data=history)
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        return PluginResult(success=False, error="No actions supported")

    def _find_history_dbs(self) -> list[Path]:
        paths = []
        home = Path.home()
        candidates = [
            home / "AppData/Local/Google/Chrome/User Data/Default/History",
            home / "AppData/Local/Microsoft/Edge/User Data/Default/History",
            home / "AppData/Roaming/Mozilla/Firefox/Profiles",
        ]
        for p in candidates:
            if p.exists():
                if p.is_dir():
                    for db in p.rglob("places.sqlite"):
                        paths.append(db)
                else:
                    paths.append(p)
        return paths

    def _read_history(self, limit: int = 20) -> list[dict]:
        # Chrome/Edge format
        for db_path in self._history_paths:
            if "places.sqlite" in str(db_path):
                continue
            try:
                import shutil, tempfile
                tmp = Path(tempfile.mkdtemp()) / "History"
                shutil.copy2(str(db_path), str(tmp))
                conn = sqlite3.connect(str(tmp))
                rows = conn.execute(
                    "SELECT url, title, last_visit_time FROM urls ORDER BY last_visit_time DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                conn.close()
                return [{"url": r[0], "title": r[1]} for r in rows]
            except Exception:
                continue
        return []

    def _search_history(self, query: str) -> list[dict]:
        for db_path in self._history_paths:
            if "places.sqlite" in str(db_path):
                continue
            try:
                import shutil, tempfile
                tmp = Path(tempfile.mkdtemp()) / "History"
                shutil.copy2(str(db_path), str(tmp))
                conn = sqlite3.connect(str(tmp))
                rows = conn.execute(
                    "SELECT url, title FROM urls WHERE title LIKE ? OR url LIKE ? ORDER BY last_visit_time DESC LIMIT 20",
                    (f"%{query}%", f"%{query}%")
                ).fetchall()
                conn.close()
                return [{"url": r[0], "title": r[1]} for r in rows]
            except Exception:
                continue
        return []
