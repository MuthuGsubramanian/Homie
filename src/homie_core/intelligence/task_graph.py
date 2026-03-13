from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any


@dataclass
class TaskNode:
    """A cluster of related user activity."""
    id: str
    apps: set[str] = field(default_factory=set)
    windows: list[dict] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    state: str = "active"
    switch_count: int = 0

    def duration_minutes(self) -> float:
        try:
            start = datetime.fromisoformat(self.first_seen)
            end = datetime.fromisoformat(self.last_seen)
            return (end - start).total_seconds() / 60
        except (ValueError, TypeError):
            return 0.0


class TaskGraph:
    def __init__(self, boundary_minutes: int = 5, stuck_minutes: int = 15,
                 stuck_switch_rate: float = 0.5):
        self._tasks: list[TaskNode] = []
        self._next_id = 0
        self._boundary = timedelta(minutes=boundary_minutes)
        self._stuck_minutes = stuck_minutes
        self._stuck_switch_rate = stuck_switch_rate
        self._last_observation: dict | None = None

    def observe(self, process: str, title: str, timestamp: str) -> None:
        obs = {"process": process, "title": title, "timestamp": timestamp}
        ts = datetime.fromisoformat(timestamp)
        matched_task = self._find_matching_task(process, title, ts)

        if matched_task:
            matched_task.windows.append(obs)
            matched_task.last_seen = timestamp
            matched_task.apps.add(process)
            if matched_task.state == "paused":
                matched_task.state = "active"
            if self._last_observation and self._last_observation["process"] != process:
                matched_task.switch_count += 1
        else:
            for t in self._tasks:
                if t.state == "active" and t.last_seen:
                    last = datetime.fromisoformat(t.last_seen)
                    if ts - last > self._boundary:
                        t.state = "paused"
            task = TaskNode(
                id=f"task_{self._next_id}", apps={process}, windows=[obs],
                first_seen=timestamp, last_seen=timestamp,
            )
            self._next_id += 1
            self._tasks.append(task)

        self._check_stuck()
        self._last_observation = obs

    def _find_matching_task(self, process: str, title: str, ts: datetime) -> TaskNode | None:
        project = self._extract_project(title)
        for task in reversed(self._tasks):
            if task.state == "completed":
                continue
            if task.last_seen:
                last = datetime.fromisoformat(task.last_seen)
                if ts - last > self._boundary and task.state != "paused":
                    continue
            if process in task.apps:
                return task
            if project:
                task_project = self._extract_project_from_task(task)
                if task_project and project.lower() == task_project.lower():
                    return task
            if task.state == "paused" and process in task.apps:
                return task
        return None

    def _extract_project(self, title: str) -> str:
        parts = title.split(" - ")
        if len(parts) >= 2:
            return parts[-1].strip()
        return ""

    def _extract_project_from_task(self, task: TaskNode) -> str:
        for w in reversed(task.windows):
            proj = self._extract_project(w["title"])
            if proj:
                return proj
        return ""

    def _check_stuck(self) -> None:
        for task in self._tasks:
            if task.state != "active":
                continue
            duration = task.duration_minutes()
            if duration >= self._stuck_minutes:
                rate = task.switch_count / max(1, duration)
                if rate >= self._stuck_switch_rate:
                    task.state = "stuck"

    def get_tasks(self) -> list[TaskNode]:
        return list(self._tasks)

    def get_incomplete_tasks(self) -> list[TaskNode]:
        return [t for t in self._tasks if t.state in ("active", "paused", "stuck")]

    def summarize(self) -> str:
        lines = []
        for t in self._tasks:
            apps = ", ".join(sorted(t.apps))
            proj = self._extract_project_from_task(t)
            dur = t.duration_minutes()
            label = f"{proj} ({apps})" if proj else apps
            lines.append(f"- [{t.state}] {label}: {dur:.0f}min, {len(t.windows)} observations")
        return "\n".join(lines) if lines else "No tasks recorded."

    def serialize(self) -> dict:
        tasks = []
        for t in self._tasks:
            tasks.append({
                "id": t.id, "apps": sorted(t.apps), "windows": t.windows,
                "first_seen": t.first_seen, "last_seen": t.last_seen,
                "state": t.state, "switch_count": t.switch_count,
            })
        return {"tasks": tasks, "next_id": self._next_id}

    @classmethod
    def deserialize(cls, data: dict) -> TaskGraph:
        tg = cls()
        tg._next_id = data.get("next_id", 0)
        for td in data.get("tasks", []):
            task = TaskNode(
                id=td["id"], apps=set(td["apps"]), windows=td["windows"],
                first_seen=td["first_seen"], last_seen=td["last_seen"],
                state=td["state"], switch_count=td["switch_count"],
            )
            tg._tasks.append(task)
        return tg
