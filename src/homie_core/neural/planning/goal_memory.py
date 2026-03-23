"""GoalMemory — persists Goal objects to a SQLite database."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .goal import Goal

logger = logging.getLogger(__name__)

CREATE_GOALS_TABLE = """\
CREATE TABLE IF NOT EXISTS goals (
    id            TEXT PRIMARY KEY,
    description   TEXT NOT NULL,
    parent_id     TEXT,
    thought_chain TEXT,
    priority      INTEGER DEFAULT 5,
    status        TEXT DEFAULT 'active',
    created_at    REAL NOT NULL,
    completed_at  REAL,
    outcome       TEXT,
    lessons_learned TEXT
);
"""


class GoalMemory:
    """Persist and query Goal objects in a SQLite ``goals`` table.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file (e.g. ``learning.db``).
        The ``goals`` table is created automatically if it does not exist.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._local = threading.local()
        # Ensure table exists on first connection
        conn = self._conn()
        conn.execute(CREATE_GOALS_TABLE)
        conn.commit()

    # -- connection management --------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    # -- public API -------------------------------------------------------

    def save_goal(self, goal: Goal) -> None:
        """Insert or replace a goal."""
        tc_json = goal.thought_chain.to_json() if goal.thought_chain else None
        status = self._derive_status(goal)
        self._conn().execute(
            """INSERT OR REPLACE INTO goals
               (id, description, parent_id, thought_chain, priority,
                status, created_at, completed_at, outcome, lessons_learned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                goal.id,
                goal.description,
                goal.parent_id,
                tc_json,
                goal.priority,
                status,
                goal.created_at,
                goal.completed_at,
                goal.outcome,
                json.dumps(goal.lessons_learned),
            ),
        )
        self._conn().commit()

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Retrieve a single goal by ID, or None."""
        row = self._conn().execute(
            "SELECT * FROM goals WHERE id = ?", (goal_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_goal(row)

    def list_active(self) -> list[Goal]:
        """Return all goals whose status is 'active', ordered by priority."""
        rows = self._conn().execute(
            "SELECT * FROM goals WHERE status = 'active' ORDER BY priority ASC, created_at ASC"
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def list_completed(self, limit: int = 50) -> list[Goal]:
        """Return the most recently completed goals."""
        rows = self._conn().execute(
            "SELECT * FROM goals WHERE status = 'completed' "
            "ORDER BY completed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def update_status(self, goal_id: str, status: str) -> None:
        """Update the status column for a goal."""
        self._conn().execute(
            "UPDATE goals SET status = ? WHERE id = ?", (status, goal_id)
        )
        self._conn().commit()

    def delete_goal(self, goal_id: str) -> None:
        """Remove a goal from the database."""
        self._conn().execute("DELETE FROM goals WHERE id = ?", (goal_id,))
        self._conn().commit()

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _derive_status(goal: Goal) -> str:
        if goal.completed_at is not None:
            return "completed"
        if goal.thought_chain and goal.thought_chain.has_failed:
            return "failed"
        return "active"

    @staticmethod
    def _row_to_goal(row: sqlite3.Row) -> Goal:
        from .goal import ThoughtChain

        tc_raw = row["thought_chain"]
        tc = ThoughtChain.from_json(tc_raw) if tc_raw else None
        lessons_raw = row["lessons_learned"]
        lessons = json.loads(lessons_raw) if lessons_raw else []
        return Goal(
            id=row["id"],
            description=row["description"],
            parent_id=row["parent_id"],
            thought_chain=tc,
            priority=row["priority"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            outcome=row["outcome"],
            lessons_learned=lessons,
        )
