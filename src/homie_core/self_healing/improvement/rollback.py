"""Rollback manager — snapshots, baselines, and auto-revert for self-modifications."""

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Union


class RollbackManager:
    """Manages file snapshots and rollbacks for self-modification safety."""

    def __init__(
        self,
        snapshot_dir: Path | str,
        evolution_dir: Path | str | None = None,
    ) -> None:
        self._snapshot_dir = Path(snapshot_dir)
        self._evolution_dir = Path(evolution_dir) if evolution_dir else self._snapshot_dir.parent / "evolution"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._evolution_dir.mkdir(parents=True, exist_ok=True)

        # {version_id: [{"original": path, "snapshot": path}]}
        self._versions: dict[str, list[dict[str, Path]]] = {}
        self._blacklist: set[str] = set()
        self._evolution_log: list[dict] = []

    def snapshot(
        self,
        files: Union[Path, str, list[Path | str]],
        reason: str = "",
    ) -> str:
        """Snapshot file(s) before modification. Returns version_id."""
        if isinstance(files, (str, Path)):
            files = [files]

        version_id = f"v-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        version_dir = self._snapshot_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        for f in files:
            f = Path(f)
            if f.exists():
                snapshot_path = version_dir / f.name
                shutil.copy2(f, snapshot_path)
                entries.append({"original": f, "snapshot": snapshot_path})

        self._versions[version_id] = entries

        # Write metadata
        meta = {
            "version_id": version_id,
            "timestamp": time.time(),
            "reason": reason,
            "files": [str(e["original"]) for e in entries],
        }
        (version_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        return version_id

    def rollback(self, version_id: str) -> None:
        """Restore files from a snapshot."""
        if version_id not in self._versions:
            raise KeyError(f"Unknown version: {version_id}")

        for entry in self._versions[version_id]:
            shutil.copy2(entry["snapshot"], entry["original"])

    def record_evolution(
        self,
        version_id: str,
        diff: str = "",
        reasoning: str = "",
        outcome: str = "",
    ) -> None:
        """Record an evolution entry for a self-modification."""
        record = {
            "version_id": version_id,
            "timestamp": time.time(),
            "diff": diff,
            "reasoning": reasoning,
            "outcome": outcome,
        }
        self._evolution_log.append(record)

        # Persist to file
        log_file = self._evolution_dir / f"{version_id}.json"
        log_file.write_text(json.dumps(record, indent=2))

    def get_evolution_log(self) -> list[dict]:
        """Return the evolution log."""
        return list(self._evolution_log)

    def blacklist(self, change_hash: str) -> None:
        """Blacklist a change to prevent re-attempting it."""
        self._blacklist.add(change_hash)

    def is_blacklisted(self, change_hash: str) -> bool:
        """Check if a change is blacklisted."""
        return change_hash in self._blacklist
