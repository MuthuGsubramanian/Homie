from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from homie_core.utils import utc_now


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source_count INTEGER NOT NULL DEFAULT 1,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                last_confirmed TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS beliefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                belief TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source_count INTEGER NOT NULL DEFAULT 1,
                context_tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS profile (
                domain TEXT PRIMARY KEY,
                data TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episodes_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                mood TEXT,
                outcome TEXT,
                context_tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS habits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                frequency TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                last_seen TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def list_tables(self) -> list[str]:
        rows = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        return [r["name"] for r in rows]

    def store_fact(self, fact: str, confidence: float = 0.5, tags: list[str] | None = None) -> int:
        now = utc_now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO semantic_memory (fact, confidence, tags, created_at, last_confirmed) VALUES (?, ?, ?, ?, ?)",
            (fact, confidence, json.dumps(tags or []), now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_facts(self, min_confidence: float = 0.0, include_archived: bool = False) -> list[dict]:
        query = "SELECT * FROM semantic_memory WHERE confidence >= ?"
        params: list[Any] = [min_confidence]
        if not include_archived:
            query += " AND archived = 0"
        rows = self._conn.execute(query, params).fetchall()
        return [{**dict(r), "tags": json.loads(r["tags"])} for r in rows]

    def store_belief(self, belief: str, confidence: float, source_count: int = 1, context_tags: list[str] | None = None) -> int:
        now = utc_now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO beliefs (belief, confidence, source_count, context_tags, created_at, last_updated) VALUES (?, ?, ?, ?, ?, ?)",
            (belief, confidence, source_count, json.dumps(context_tags or []), now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_beliefs(self, min_confidence: float = 0.0) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM beliefs WHERE confidence >= ?", (min_confidence,)).fetchall()
        return [{**dict(r), "context_tags": json.loads(r["context_tags"])} for r in rows]

    def record_feedback(self, channel: str, content: str, context: dict | None = None) -> int:
        now = utc_now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO feedback (channel, content, context, created_at) VALUES (?, ?, ?, ?)",
            (channel, content, json.dumps(context or {}), now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_recent_feedback(self, limit: int = 50, channel: str | None = None) -> list[dict]:
        query = "SELECT * FROM feedback"
        params: list[Any] = []
        if channel:
            query += " WHERE channel = ?"
            params.append(channel)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [{**dict(r), "context": json.loads(r["context"])} for r in rows]

    def store_profile(self, domain: str, data: dict) -> None:
        now = utc_now().isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO profile (domain, data, updated_at) VALUES (?, ?, ?)",
            (domain, json.dumps(data), now),
        )
        self._conn.commit()

    def get_profile(self, domain: str) -> dict | None:
        row = self._conn.execute("SELECT data FROM profile WHERE domain = ?", (domain,)).fetchone()
        if row:
            return json.loads(row["data"])
        return None

    def record_episode_meta(self, summary: str, mood: str | None = None, outcome: str | None = None, context_tags: list[str] | None = None) -> int:
        now = utc_now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO episodes_meta (summary, mood, outcome, context_tags, created_at) VALUES (?, ?, ?, ?, ?)",
            (summary, mood, outcome, json.dumps(context_tags or []), now),
        )
        self._conn.commit()
        return cur.lastrowid

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
