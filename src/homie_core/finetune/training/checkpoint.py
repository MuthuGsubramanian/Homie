"""Checkpoint manager for training pause/resume across finetuning cycles."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


class CheckpointManager:
    """Manages training checkpoints so cycles can be paused and resumed."""

    def __init__(self, base_dir: Path, keep_recent: int = 3):
        self._base_dir = Path(base_dir)
        self._keep_recent = keep_recent

    def save(self, state: dict, cycle: int) -> None:
        """Persist *state* dict as JSON under ``cycle-{cycle}/checkpoint.json``."""
        path = self._base_dir / f"cycle-{cycle}" / "checkpoint.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))

    def load(self, cycle: int) -> dict | None:
        """Return the saved state for *cycle*, or ``None`` if absent."""
        path = self._base_dir / f"cycle-{cycle}" / "checkpoint.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def has_checkpoint(self, cycle: int) -> bool:
        """Return whether a checkpoint exists for *cycle*."""
        return (self._base_dir / f"cycle-{cycle}" / "checkpoint.json").exists()

    def cleanup(self) -> None:
        """Delete checkpoint dirs older than *keep_recent*."""
        cycle_dirs: list[tuple[int, Path]] = []
        for child in self._base_dir.iterdir():
            if child.is_dir():
                match = re.match(r"^cycle-(\d+)$", child.name)
                if match:
                    cycle_dirs.append((int(match.group(1)), child))

        cycle_dirs.sort(key=lambda t: t[0])

        if len(cycle_dirs) > self._keep_recent:
            to_delete = cycle_dirs[: len(cycle_dirs) - self._keep_recent]
            for _, path in to_delete:
                shutil.rmtree(path)
