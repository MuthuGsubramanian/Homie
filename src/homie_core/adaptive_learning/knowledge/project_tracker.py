"""Project tracker — builds lightweight knowledge graph of user's projects."""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from ..storage import LearningStorage


@dataclass
class ProjectInfo:
    name: str
    path: str
    branch: str = "main"
    recent_activity: list[str] = field(default_factory=list)
    registered_at: float = field(default_factory=time.time)


class ProjectTracker:
    """Tracks projects and builds knowledge about them."""

    def __init__(self, storage: LearningStorage, max_activity: int = 20) -> None:
        self._storage = storage
        self._max_activity = max_activity
        self._lock = threading.Lock()
        self._projects: dict[str, ProjectInfo] = {}
        self._knowledge: dict[str, list[dict]] = {}  # {subject: [{predicate, object}]}
        self._active: Optional[str] = None

    def register_project(self, name: str, path: str, branch: str = "main") -> None:
        """Register a project."""
        with self._lock:
            self._projects[name] = ProjectInfo(name=name, path=path, branch=branch)

    def list_projects(self) -> list[str]:
        """List registered project names."""
        with self._lock:
            return list(self._projects.keys())

    def get_project(self, name: str) -> Optional[dict]:
        """Get project info as dict."""
        with self._lock:
            info = self._projects.get(name)
            if info is None:
                return None
            return {
                "name": info.name,
                "path": info.path,
                "branch": info.branch,
                "recent_activity": list(info.recent_activity),
            }

    def update_activity(self, name: str, activity: str) -> None:
        """Record recent activity for a project."""
        with self._lock:
            info = self._projects.get(name)
            if info:
                info.recent_activity.append(activity)
                if len(info.recent_activity) > self._max_activity:
                    info.recent_activity = info.recent_activity[-self._max_activity:]

    def add_knowledge(self, subject: str, predicate: str, obj: str) -> None:
        """Add a knowledge triple (subject, predicate, object)."""
        with self._lock:
            if subject not in self._knowledge:
                self._knowledge[subject] = []
            self._knowledge[subject].append({"predicate": predicate, "object": obj})

    def get_knowledge(self, subject: str) -> list[dict]:
        """Get knowledge triples for a subject."""
        with self._lock:
            return list(self._knowledge.get(subject, []))

    def set_active(self, name: str) -> None:
        """Set the currently active project."""
        self._active = name

    @property
    def active_project(self) -> Optional[str]:
        return self._active
