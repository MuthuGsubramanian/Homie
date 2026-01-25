from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from homie.utils import timestamp


@dataclass
class WorkflowSession:
    id: str
    name: str
    permissions: Dict[str, bool]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    storage_path: Optional[Path] = None


class WorkflowRecorder:
    """Manual workflow recorder. No passive capture; only when explicitly started."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.active: Optional[WorkflowSession] = None

    def start(self, name: str, permissions: Dict[str, bool]) -> WorkflowSession:
        session = WorkflowSession(
            id=str(abs(hash(name + timestamp()))),
            name=name,
            permissions=permissions,
        )
        self.active = session
        return session

    def checkpoint(self, note: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.active:
            return
        self.active.steps.append(
            {
                "ts": timestamp(),
                "kind": "checkpoint",
                "note": note,
                "payload": payload or {},
            }
        )

    def stop(self) -> Optional[WorkflowSession]:
        if not self.active:
            return None
        session = self.active
        session.storage_path = self.root / f"workflow_{session.id}.json"
        session.storage_path.write_text(
            json.dumps(
                {
                    "id": session.id,
                    "name": session.name,
                    "permissions": session.permissions,
                    "steps": session.steps,
                    "ended_at": timestamp(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.active = None
        return session


__all__ = ["WorkflowRecorder", "WorkflowSession"]
