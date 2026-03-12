"""Scheduled task system for the Homie AI assistant.

Provides cron-like job scheduling with persistent storage, supporting
duration-based delays, recurring intervals, standard cron expressions,
ISO timestamps, and human-readable shortcuts. Designed to be ticked
every 60 seconds from the daemon's main loop.

Inspired by hermes-agent's cron module.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Schedule parsing
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+)\s*([mhd])$", re.IGNORECASE)
_INTERVAL_RE = re.compile(r"^every\s+(\d+)\s*([mhd])$", re.IGNORECASE)
_CRON_RE = re.compile(
    r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$"
)
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")

_HUMAN_SHORTCUTS = {
    "hourly": "0 * * * *",
    "daily": "0 0 * * *",
    "weekly": "0 0 * * 0",
}

_UNIT_MAP = {"m": "minutes", "h": "hours", "d": "days"}


def parse_schedule(spec: str) -> dict:
    """Parse a human-friendly schedule specification into a structured dict.

    Supported formats:
        - Duration-based (one-shot after delay): ``"30m"``, ``"2h"``, ``"1d"``
        - Interval-based (recurring): ``"every 30m"``, ``"every 2h"``
        - Cron expressions (5-field): ``"0 9 * * *"``
        - ISO timestamps (one-shot): ``"2026-03-12T14:00"``
        - Human-readable shortcuts: ``"daily"``, ``"hourly"``, ``"weekly"``

    Returns:
        A dict with a ``"type"`` key (one of ``"duration"``, ``"interval"``,
        ``"cron"``, ``"iso"``, ``"cron"`` for shortcuts) and associated data.

    Raises:
        ValueError: If *spec* does not match any known format.
    """
    spec = spec.strip()
    lower = spec.lower()

    # Human-readable shortcuts -> cron
    if lower in _HUMAN_SHORTCUTS:
        return {"type": "cron", "expression": _HUMAN_SHORTCUTS[lower]}

    # Interval: "every 30m"
    m = _INTERVAL_RE.match(spec)
    if m:
        amount, unit = int(m.group(1)), m.group(2).lower()
        return {
            "type": "interval",
            "seconds": int(timedelta(**{_UNIT_MAP[unit]: amount}).total_seconds()),
        }

    # Duration (one-shot): "30m"
    m = _DURATION_RE.match(spec)
    if m:
        amount, unit = int(m.group(1)), m.group(2).lower()
        return {
            "type": "duration",
            "seconds": int(timedelta(**{_UNIT_MAP[unit]: amount}).total_seconds()),
        }

    # ISO timestamp
    if _ISO_RE.match(spec):
        dt = datetime.fromisoformat(spec)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return {"type": "iso", "datetime": dt.isoformat()}

    # Cron expression (5 fields)
    if _CRON_RE.match(spec):
        _validate_cron_fields(spec)
        return {"type": "cron", "expression": spec}

    raise ValueError(f"Unrecognised schedule specification: {spec!r}")


def _validate_cron_fields(expr: str) -> None:
    """Basic sanity check on a 5-field cron expression."""
    fields = expr.split()
    limits = [
        (0, 59),   # minute
        (0, 23),   # hour
        (1, 31),   # day of month
        (1, 12),   # month
        (0, 6),    # day of week
    ]
    for tok, (lo, hi) in zip(fields, limits):
        if tok == "*":
            continue
        # Handle */N step values
        if tok.startswith("*/"):
            step = tok[2:]
            if not step.isdigit() or int(step) < 1:
                raise ValueError(f"Invalid cron step: {tok}")
            continue
        # Handle ranges like 1-5
        if "-" in tok:
            parts = tok.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                a, b = int(parts[0]), int(parts[1])
                if a < lo or b > hi or a > b:
                    raise ValueError(f"Cron range out of bounds: {tok}")
                continue
        # Handle comma-separated lists
        for part in tok.split(","):
            if not part.isdigit():
                raise ValueError(f"Invalid cron token: {tok}")
            val = int(part)
            if val < lo or val > hi:
                raise ValueError(f"Cron value {val} out of range [{lo}, {hi}]")


# ---------------------------------------------------------------------------
# Next-run computation helpers
# ---------------------------------------------------------------------------

def _next_run_from_parsed(parsed: dict, from_dt: datetime) -> datetime:
    """Compute the next run time given a parsed schedule and a reference time."""
    stype = parsed["type"]

    if stype == "duration":
        return from_dt + timedelta(seconds=parsed["seconds"])

    if stype == "interval":
        return from_dt + timedelta(seconds=parsed["seconds"])

    if stype == "iso":
        return datetime.fromisoformat(parsed["datetime"])

    if stype == "cron":
        return _next_cron_time(parsed["expression"], from_dt)

    raise ValueError(f"Unknown schedule type: {stype}")


def _next_cron_time(expression: str, from_dt: datetime) -> datetime:
    """Find the next datetime matching a 5-field cron expression.

    Uses a brute-force minute-by-minute scan capped at ~2 years to avoid
    infinite loops. This is perfectly adequate when the scheduler ticks
    every 60 seconds.
    """
    fields = expression.split()
    matchers = [
        _cron_field_matcher(fields[0], 0, 59),   # minute
        _cron_field_matcher(fields[1], 0, 23),    # hour
        _cron_field_matcher(fields[2], 1, 31),    # day-of-month
        _cron_field_matcher(fields[3], 1, 12),    # month
        _cron_field_matcher(fields[4], 0, 6),     # day-of-week
    ]

    # Start from the next whole minute after from_dt
    candidate = from_dt.replace(second=0, microsecond=0) + timedelta(minutes=1)
    max_iterations = 525960  # ~1 year of minutes
    for _ in range(max_iterations):
        if (
            candidate.minute in matchers[0]
            and candidate.hour in matchers[1]
            and candidate.day in matchers[2]
            and candidate.month in matchers[3]
            and candidate.weekday() in _convert_dow(matchers[4])
        ):
            return candidate
        candidate += timedelta(minutes=1)

    raise RuntimeError(f"Could not resolve next run for cron: {expression}")


def _cron_field_matcher(token: str, lo: int, hi: int) -> set[int]:
    """Expand a single cron field token into a set of matching integers."""
    if token == "*":
        return set(range(lo, hi + 1))
    if token.startswith("*/"):
        step = int(token[2:])
        return {v for v in range(lo, hi + 1) if (v - lo) % step == 0}
    result: set[int] = set()
    for part in token.split(","):
        if "-" in part:
            a, b = part.split("-")
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(part))
    return result


def _convert_dow(cron_dow_set: set[int]) -> set[int]:
    """Convert cron day-of-week values (0=Sunday) to Python weekday (0=Monday)."""
    mapping = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    return {mapping[d] for d in cron_dow_set if d in mapping}


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """A single scheduled task."""

    id: str
    name: str
    prompt: str
    schedule: str
    created_at: str
    next_run: str
    last_run: Optional[str] = None
    repeat_count: int = 0
    max_repeats: Optional[int] = None  # None = infinite
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        """Deserialise from a plain dict."""
        return cls(**data)


# ---------------------------------------------------------------------------
# Portable file locking
# ---------------------------------------------------------------------------

def _lock_file(fp):
    """Acquire an exclusive lock on an open file handle (cross-platform)."""
    if sys.platform == "win32":
        import msvcrt
        msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(fp):
    """Release an exclusive lock on an open file handle (cross-platform)."""
    if sys.platform == "win32":
        import msvcrt
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fp, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# JobStore
# ---------------------------------------------------------------------------

class JobStore:
    """Persistent, JSON-backed store for scheduled jobs.

    Jobs are kept in a single JSON file with atomic writes (write to a
    temporary file, then rename) and restrictive file permissions (0o600).

    Args:
        path: Filesystem path for the jobs JSON file.  Defaults to
              ``~/.homie/scheduler/jobs.json``.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.path.join(Path.home(), ".homie", "scheduler", "jobs.json")
        self.path = Path(path)
        self._jobs: dict[str, Job] = {}
        self.load()

    # -- persistence --------------------------------------------------------

    def load(self) -> None:
        """Load jobs from disk.  Missing or empty files are silently ignored."""
        if not self.path.exists():
            self._jobs = {}
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._jobs = {j["id"]: Job.from_dict(j) for j in data}
        except (json.JSONDecodeError, KeyError):
            self._jobs = {}

    def save(self) -> None:
        """Persist current jobs to disk with an atomic write."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(
            [j.to_dict() for j in self._jobs.values()],
            indent=2,
            default=str,
        )

        # Atomic write: temp file -> rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                fp.write(payload)

            # Set restrictive permissions (best-effort on Windows)
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass

            # On Windows, the target must not exist for os.rename.
            if sys.platform == "win32" and self.path.exists():
                os.replace(tmp_path, str(self.path))
            else:
                os.rename(tmp_path, str(self.path))
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # -- CRUD ---------------------------------------------------------------

    def create_job(
        self,
        name: str,
        prompt: str,
        schedule: str,
        max_repeats: Optional[int] = None,
        tags: Optional[list[str]] = None,
    ) -> Job:
        """Create a new job, persist it, and return it.

        Args:
            name:        Human-readable name for the job.
            prompt:      The text/instruction to send to Homie when the job fires.
            schedule:    A schedule spec accepted by :func:`parse_schedule`.
            max_repeats: Maximum number of executions (``None`` for infinite).
            tags:        Optional list of string tags for organisation.

        Returns:
            The newly created :class:`Job`.
        """
        parsed = parse_schedule(schedule)
        now = datetime.now(timezone.utc)
        next_run = _next_run_from_parsed(parsed, now)

        job = Job(
            id=str(uuid.uuid4()),
            name=name,
            prompt=prompt,
            schedule=schedule,
            created_at=now.isoformat(),
            next_run=next_run.isoformat(),
            max_repeats=max_repeats,
            tags=tags or [],
        )
        self._jobs[job.id] = job
        self.save()
        return job

    def delete_job(self, job_id: str) -> bool:
        """Remove a job by ID.  Returns ``True`` if it existed."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self.save()
            return True
        return False

    def get_due_jobs(self) -> list[Job]:
        """Return all enabled jobs whose *next_run* is at or before now."""
        now = datetime.now(timezone.utc)
        due: list[Job] = []
        for job in self._jobs.values():
            if not job.enabled:
                continue
            next_run = datetime.fromisoformat(job.next_run)
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=timezone.utc)
            if next_run <= now:
                due.append(job)
        return due

    def mark_completed(self, job_id: str, output: str) -> None:
        """Record that a job has executed and compute its next run.

        If *max_repeats* is set and the repeat count reaches it, the job is
        automatically deleted.

        Args:
            job_id: The UUID of the completed job.
            output:  A summary string of the execution result (stored for
                     reference but not persisted beyond *last_run* timestamp).
        """
        job = self._jobs.get(job_id)
        if job is None:
            return

        now = datetime.now(timezone.utc)
        job.last_run = now.isoformat()
        job.repeat_count += 1

        # Auto-delete if max repeats reached
        if job.max_repeats is not None and job.repeat_count >= job.max_repeats:
            del self._jobs[job_id]
            self.save()
            return

        # Compute next run
        parsed = parse_schedule(job.schedule)
        if parsed["type"] in ("duration", "iso"):
            # One-shot schedules: disable after single execution
            job.enabled = False
        else:
            job.next_run = _next_run_from_parsed(parsed, now).isoformat()

        self.save()

    def list_jobs(self) -> list[Job]:
        """Return all jobs (enabled and disabled)."""
        return list(self._jobs.values())


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Coordinates job execution on a regular tick cycle.

    Should be called every ~60 seconds from the Homie daemon's main loop.
    Uses a file-based lock to prevent concurrent ticks across processes.

    Args:
        job_store:  A :class:`JobStore` instance for reading/writing jobs.
        on_job_due: Callback invoked with a :class:`Job` when it is due.
                    Must return a string summarising the execution result.
    """

    def __init__(
        self,
        job_store: JobStore,
        on_job_due: Callable[[Job], str],
    ) -> None:
        self.job_store = job_store
        self.on_job_due = on_job_due
        self._lock_path = self.job_store.path.parent / ".tick.lock"

    def tick(self) -> list[tuple[Job, str]]:
        """Check for due jobs, execute them, and return results.

        Acquires a file-based lock so that only one tick runs at a time.
        If the lock cannot be acquired (another tick is in progress), an
        empty list is returned immediately.

        Returns:
            A list of ``(job, output)`` tuples for every job that executed
            during this tick.
        """
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            lock_fp = open(self._lock_path, "w", encoding="utf-8")
            _lock_file(lock_fp)
        except (OSError, PermissionError):
            # Another tick is in progress
            return []

        try:
            # Reload from disk to pick up external changes
            self.job_store.load()
            due = self.job_store.get_due_jobs()
            results: list[tuple[Job, str]] = []

            for job in due:
                try:
                    output = self.on_job_due(job)
                except Exception as exc:
                    output = f"Error: {exc}"

                self.job_store.mark_completed(job.id, output)
                results.append((job, output))

            return results
        finally:
            try:
                _unlock_file(lock_fp)
            except OSError:
                pass
            lock_fp.close()
