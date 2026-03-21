"""SessionPersistence — save and restore working memory state across sessions.

Only a lightweight summary is persisted (not the full conversation buffer)
so the file stays small and loads quickly on startup.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from homie_core.memory.working import WorkingMemory

# Keys that are cheap to restore and useful for context continuity
_PERSIST_KEYS = (
    "active_window",
    "active_process",
    "activity_type",
    "flow_score",
    "task_description",
    "sentiment",
)

# Keys that are restored into WorkingMemory on startup
_RESTORE_KEYS = ("activity_type", "task_description")


class SessionPersistence:
    """Save and restore session context across restarts."""

    def __init__(self, storage_path: str | Path) -> None:
        self._path = Path(storage_path) / "last_session.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, working_memory: WorkingMemory) -> None:
        """Serialize the relevant parts of *working_memory* to disk."""
        state: dict = {}

        for key in _PERSIST_KEYS:
            val = working_memory.get(key)
            if val is not None:
                state[key] = val

        # Save a digest of the conversation — not the full buffer
        conversation = working_memory.get_conversation()
        if conversation:
            state["last_conversation_length"] = len(conversation)
            last_content = conversation[-1].get("content", "") if conversation else ""
            state["last_message"] = last_content[:200]

        state["saved_at"] = datetime.now().isoformat()

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(state, indent=2, default=str))

    def restore(self, working_memory: WorkingMemory) -> dict:
        """Load the last saved session state and apply lightweight keys to *working_memory*.

        Returns the full state dict (including metadata keys), or ``{}`` if
        no session file exists or it cannot be parsed.
        """
        if not self._path.exists():
            return {}

        try:
            state: dict = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError, ValueError):
            return {}

        for key in _RESTORE_KEYS:
            if key in state:
                working_memory.update(key, state[key])

        return state

    def exists(self) -> bool:
        """Return True if a persisted session file is present."""
        return self._path.exists()
