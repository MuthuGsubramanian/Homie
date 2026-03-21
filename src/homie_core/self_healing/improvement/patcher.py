"""Code patcher — generates and applies source code modifications."""

import logging
from pathlib import Path
from typing import Optional

from .rollback import RollbackManager

logger = logging.getLogger(__name__)


class CodePatcher:
    """Applies targeted code patches to source files with rollback support."""

    def __init__(
        self,
        rollback_manager: RollbackManager,
        project_root: Path | str,
        locked_paths: Optional[list[str]] = None,
    ) -> None:
        self._rollback = rollback_manager
        self._root = Path(project_root)
        self._locked = locked_paths or []

    def _is_locked(self, file_path: Path) -> bool:
        """Check if a file is in the core lock list."""
        try:
            rel = file_path.relative_to(self._root)
        except ValueError:
            rel = file_path
        rel_str = str(rel).replace("\\", "/")
        for lock in self._locked:
            if lock.endswith("/"):
                if rel_str.startswith(lock) or rel_str.startswith(lock.rstrip("/")):
                    return True
            elif rel_str == lock:
                return True
        return False

    def apply_patch(
        self,
        file_path: Path | str,
        old_text: str,
        new_text: str,
        reason: str = "",
    ) -> str:
        """Apply a text replacement patch to a file.

        Returns version_id for rollback.
        Raises PermissionError if file is core-locked.
        Raises ValueError if old_text not found in file.
        """
        file_path = Path(file_path)

        if self._is_locked(file_path):
            raise PermissionError(f"File is core-locked: {file_path}")

        content = file_path.read_text()
        if old_text not in content:
            raise ValueError(f"Target text not found in {file_path}")

        # Snapshot before modification
        version_id = self._rollback.snapshot(file_path, reason=reason)

        # Apply patch
        new_content = content.replace(old_text, new_text, 1)
        file_path.write_text(new_content)

        logger.info("Applied patch to %s (version: %s): %s", file_path, version_id, reason)
        return version_id
