from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Schema aligned with vNext spec
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS machines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT,
    ip TEXT UNIQUE NOT NULL,
    os_type TEXT,
    tailnet_id TEXT,
    capabilities_json TEXT,
    last_seen_ts TEXT,
    status TEXT,
    quiet_hours_json TEXT,
    blocked INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER,
    command TEXT,
    user TEXT,
    reason TEXT,
    status TEXT,
    stdout_path TEXT,
    stderr_path TEXT,
    started_at TEXT,
    finished_at TEXT,
    rollback_plan_json TEXT,
    autonomy_level TEXT,
    risk_class TEXT,
    created_by TEXT,
    updated_at TEXT,
    target_ip TEXT,
    FOREIGN KEY(machine_id) REFERENCES machines(id)
);
CREATE TABLE IF NOT EXISTS suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER,
    intent_tag TEXT,
    summary TEXT,
    details_json TEXT,
    confidence REAL,
    signals_json TEXT,
    alternatives_json TEXT,
    autonomy_level_offered TEXT,
    status TEXT,
    created_at TEXT,
    executed_run_id INTEGER,
    FOREIGN KEY(machine_id) REFERENCES machines(id)
);
CREATE TABLE IF NOT EXISTS workflows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    scope TEXT,
    redaction_rules_json TEXT,
    recording_path TEXT,
    created_at TEXT,
    last_run_at TEXT,
    success_rate REAL,
    approval_required INTEGER,
    owner TEXT,
    tags TEXT
);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    actor TEXT,
    machine_id INTEGER,
    type TEXT,
    ref_id TEXT,
    what TEXT,
    why TEXT,
    signals_json TEXT,
    confidence REAL,
    outcome TEXT,
    rollback_available INTEGER,
    autonomy_level TEXT,
    risk_class TEXT,
    FOREIGN KEY(machine_id) REFERENCES machines(id)
);
CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE,
    value_json TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS autonomy_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER,
    task_type TEXT,
    level TEXT,
    decision_ts TEXT,
    reason TEXT,
    success INTEGER,
    notes TEXT,
    FOREIGN KEY(machine_id) REFERENCES machines(id)
);
CREATE TABLE IF NOT EXISTS knowledge_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER,
    intent_tag TEXT,
    symptoms_json TEXT,
    fix_steps_json TEXT,
    evidence_json TEXT,
    created_at TEXT,
    source_run_id INTEGER,
    fingerprint_hash TEXT,
    rollout_status TEXT,
    FOREIGN KEY(machine_id) REFERENCES machines(id)
);
CREATE INDEX IF NOT EXISTS idx_runs_machine ON runs(machine_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_machine ON suggestions(machine_id);
CREATE INDEX IF NOT EXISTS idx_ledger_machine ON ledger(machine_id);
"""


class Storage:
    """SQLite adapter supporting HOMIE controller state."""

    def __init__(self, path: Path):
        self.path = path.expanduser()
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # Machine helpers
    def upsert_machine(self, ip: str, display_name: Optional[str] = None, os_type: Optional[str] = None,
                      tailnet_id: Optional[str] = None, capabilities: Optional[Dict[str, Any]] = None,
                      status: str = "unknown", last_seen_ts: Optional[str] = None) -> int:
        caps_json = json.dumps(capabilities or {})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO machines(display_name, ip, os_type, tailnet_id, capabilities_json, last_seen_ts, status)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(ip) DO UPDATE SET
                    display_name=excluded.display_name,
                    os_type=excluded.os_type,
                    tailnet_id=excluded.tailnet_id,
                    capabilities_json=excluded.capabilities_json,
                    last_seen_ts=excluded.last_seen_ts,
                    status=excluded.status
                """,
                (display_name, ip, os_type, tailnet_id, caps_json, last_seen_ts, status),
            )
            row = conn.execute("SELECT id FROM machines WHERE ip=?", (ip,)).fetchone()
        return int(row["id"]) if row else -1

    def _machine_id(self, ip: str) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM machines WHERE ip=?", (ip,)).fetchone()
            return int(row["id"]) if row else None

    # Runs
    def record_run(
        self,
        machine_ip: str,
        command: str,
        user: str,
        reason: str,
        status: str,
        autonomy_level: str,
        risk_class: str,
        rollback_plan: Optional[Dict[str, Any]] = None,
        started_at: Optional[str] = None,
    ) -> int:
        machine_id = self._machine_id(machine_ip) or self.upsert_machine(machine_ip)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(machine_id, command, user, reason, status, rollback_plan_json,
                                 autonomy_level, risk_class, created_by, started_at, target_ip, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
                """,
                (
                    machine_id,
                    command,
                    user,
                    reason,
                    status,
                    json.dumps(rollback_plan or {}),
                    autonomy_level,
                    risk_class,
                    "controller",
                    started_at,
                    machine_ip,
                ),
            )
            row = conn.execute("SELECT last_insert_rowid()").fetchone()
        return int(row[0])

    def complete_run(
        self,
        run_id: int,
        status: str,
        exit_status: Optional[int],
        stdout_path: Optional[str],
        stderr_path: Optional[str],
        finished_at: Optional[str],
        error: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status=?, stdout_path=?, stderr_path=?, finished_at=?, updated_at=datetime('now')
                WHERE id=?
                """,
                (status if error is None else "failed", stdout_path, stderr_path, finished_at, run_id),
            )

    # Suggestions
    def record_suggestion(
        self,
        machine_ip: str,
        intent_tag: str,
        summary: str,
        details: Dict[str, Any],
        confidence: float,
        signals: Dict[str, Any],
        alternatives: Dict[str, Any],
        autonomy_level_offered: str,
        status: str,
        created_at: str,
    ) -> int:
        machine_id = self._machine_id(machine_ip) or self.upsert_machine(machine_ip)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO suggestions(machine_id,intent_tag,summary,details_json,confidence,
                                        signals_json,alternatives_json,autonomy_level_offered,status,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    machine_id,
                    intent_tag,
                    summary,
                    json.dumps(details or {}),
                    confidence,
                    json.dumps(signals or {}),
                    json.dumps(alternatives or {}),
                    autonomy_level_offered,
                    status,
                    created_at,
                ),
            )
            row = conn.execute("SELECT last_insert_rowid()").fetchone()
        return int(row[0])

    # Ledger
    def record_ledger(
        self,
        ts: str,
        actor: str,
        machine_ip: str,
        entry_type: str,
        ref_id: str,
        what: str,
        why: str,
        signals: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        outcome: Optional[str] = None,
        rollback_available: bool = False,
        autonomy_level: Optional[str] = None,
        risk_class: Optional[str] = None,
    ) -> None:
        machine_id = self._machine_id(machine_ip) or self.upsert_machine(machine_ip)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ledger(ts, actor, machine_id, type, ref_id, what, why, signals_json,
                                   confidence, outcome, rollback_available, autonomy_level, risk_class)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ts,
                    actor,
                    machine_id,
                    entry_type,
                    ref_id,
                    what,
                    why,
                    json.dumps(signals or {}),
                    confidence,
                    outcome,
                    int(bool(rollback_available)),
                    autonomy_level,
                    risk_class,
                ),
            )

    # Preferences
    def set_preference(self, key: str, value: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO preferences(key, value_json, updated_at)
                VALUES(?,?,datetime('now'))
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=datetime('now')
                """,
                (key, json.dumps(value)),
            )

    def get_preference(self, key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM preferences WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["value_json"])
        except Exception:  # noqa: BLE001
            return None

    # Autonomy history
    def record_autonomy(self, machine_ip: str, task_type: str, level: str, reason: str, success: bool) -> None:
        machine_id = self._machine_id(machine_ip) or self.upsert_machine(machine_ip)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO autonomy_history(machine_id, task_type, level, decision_ts, reason, success)
                VALUES(?,?,?,datetime('now'),?,?)
                """,
                (machine_id, task_type, level, reason, int(bool(success))),
            )

    # Knowledge index
    def add_knowledge_item(
        self,
        machine_ip: Optional[str],
        intent_tag: str,
        symptoms: Dict[str, Any],
        fix_steps: Dict[str, Any],
        evidence: Dict[str, Any],
        source_run_id: Optional[int],
        fingerprint_hash: str,
        rollout_status: str = "candidate",
    ) -> int:
        machine_id = self._machine_id(machine_ip) if machine_ip else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_index(machine_id,intent_tag,symptoms_json,fix_steps_json,
                                            evidence_json,created_at,source_run_id,fingerprint_hash,rollout_status)
                VALUES(?,?,?,?,?,datetime('now'),?,?,?)
                """,
                (
                    machine_id,
                    intent_tag,
                    json.dumps(symptoms or {}),
                    json.dumps(fix_steps or {}),
                    json.dumps(evidence or {}),
                    source_run_id,
                    fingerprint_hash,
                    rollout_status,
                ),
            )
            row = conn.execute("SELECT last_insert_rowid()").fetchone()
        return int(row[0])

    # Retrieval helpers
    def recent_runs(self, limit: int = 50) -> Iterable[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT runs.*, machines.ip as machine_ip, machines.display_name
                FROM runs
                LEFT JOIN machines ON runs.machine_id = machines.id
                ORDER BY runs.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_suggestions(self, limit: int = 50) -> Iterable[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT suggestions.*, machines.ip as machine_ip, machines.display_name
                FROM suggestions
                LEFT JOIN machines ON suggestions.machine_id = machines.id
                ORDER BY suggestions.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_ledger(self, limit: int = 100) -> Iterable[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ledger.*, machines.ip as machine_ip, machines.display_name
                FROM ledger
                LEFT JOIN machines ON ledger.machine_id = machines.id
                ORDER BY ledger.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_machines(self) -> Iterable[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, display_name, ip, os_type, tailnet_id, capabilities_json, last_seen_ts, status, blocked
                FROM machines ORDER BY id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]


__all__ = ["Storage"]
