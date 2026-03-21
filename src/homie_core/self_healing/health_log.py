"""SQLite-backed health event log."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .event_bus import HealthEvent

_SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}


class HealthLog:
    """Persistent health event log backed by SQLite."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Create the database and health_events table."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS health_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                module TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                details TEXT NOT NULL,
                version_id TEXT DEFAULT ''
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_module ON health_events(module)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_type ON health_events(event_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_ts ON health_events(timestamp)
        """)
        self._conn.commit()

    def write(self, event: HealthEvent) -> None:
        """Write a health event to the log."""
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT INTO health_events (timestamp, module, event_type, severity, details, version_id) VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.timestamp,
                event.module,
                event.event_type,
                event.severity,
                json.dumps(event.details),
                event.version_id,
            ),
        )
        self._conn.commit()

    def query(
        self,
        module: Optional[str] = None,
        event_type: Optional[str] = None,
        min_severity: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query health events with optional filters."""
        if self._conn is None:
            return []

        clauses = []
        params: list = []

        if module:
            clauses.append("module = ?")
            params.append(module)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if min_severity and min_severity in _SEVERITY_ORDER:
            min_level = _SEVERITY_ORDER[min_severity]
            allowed = [s for s, level in _SEVERITY_ORDER.items() if level >= min_level]
            placeholders = ",".join("?" for _ in allowed)
            clauses.append(f"severity IN ({placeholders})")
            params.extend(allowed)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM health_events WHERE {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def cleanup(self, max_age_days: int = 30) -> int:
        """Delete events older than max_age_days. Returns count deleted."""
        if self._conn is None:
            return 0
        cutoff = time.time() - (max_age_days * 86400)
        cursor = self._conn.execute(
            "DELETE FROM health_events WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
