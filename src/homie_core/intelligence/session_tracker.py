from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from homie_core.intelligence.task_graph import TaskGraph
from homie_core.utils import utc_now


class SessionTracker:
    """Persists task graph and session metadata across sessions."""

    def __init__(self, storage_dir: Path | str):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._session_file = self._dir / "last_session.json"

    def save_session(self, task_graph: TaskGraph,
                     apps_used: dict[str, float] | None = None) -> None:
        data = {
            "task_graph": task_graph.serialize(),
            "apps_used": apps_used or {},
            "saved_at": utc_now().isoformat(),
        }
        self._session_file.write_text(json.dumps(data, indent=2))

    def load_last_session(self) -> Optional[dict[str, Any]]:
        if not self._session_file.exists():
            return None
        try:
            data = json.loads(self._session_file.read_text())
            data["task_graph"] = TaskGraph.deserialize(data["task_graph"])
            return data
        except (json.JSONDecodeError, KeyError):
            return None

    def get_resumption_summary(self) -> Optional[str]:
        session = self.load_last_session()
        if not session:
            return None

        tg: TaskGraph = session["task_graph"]
        apps = session.get("apps_used", {})
        incomplete = tg.get_incomplete_tasks()

        lines = []
        if incomplete:
            lines.append("You left off with these tasks:")
            for t in incomplete:
                proj = tg._extract_project_from_task(t)
                app_list = ", ".join(sorted(t.apps))
                label = proj if proj else app_list
                lines.append(f"  - [{t.state}] {label} ({len(t.windows)} activities)")

        if apps:
            top = sorted(apps.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("\nApps used yesterday:")
            for app, secs in top:
                mins = secs / 60
                lines.append(f"  - {app}: {mins:.0f} min")

        return "\n".join(lines) if lines else None

    def generate_digest(self, task_graph: TaskGraph,
                        apps_used: dict[str, float] | None = None,
                        switch_count: int = 0) -> str:
        apps = apps_used or {}
        total_hours = sum(apps.values()) / 3600 if apps else 0

        lines = ["End-of-day summary:"]
        lines.append(f"  Total active time: {total_hours:.1f} hours")
        lines.append(f"  Context switches: {switch_count}")

        if apps:
            lines.append("  Top apps:")
            top = sorted(apps.items(), key=lambda x: x[1], reverse=True)[:5]
            for app, secs in top:
                lines.append(f"    - {app}: {secs / 60:.0f} min")

        tasks = task_graph.get_tasks()
        if tasks:
            lines.append(f"  Tasks tracked: {len(tasks)}")
            stuck = [t for t in tasks if t.state == "stuck"]
            if stuck:
                lines.append(f"  Stuck tasks: {len(stuck)}")

        incomplete = task_graph.get_incomplete_tasks()
        if incomplete:
            lines.append(f"  Incomplete tasks: {len(incomplete)}")

        return "\n".join(lines)
