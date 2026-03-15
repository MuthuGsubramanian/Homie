from __future__ import annotations

import fnmatch
import re
from typing import Optional

from homie_core.backend.protocol import (
    EditResult,
    FileContent,
    FileInfo,
    GrepMatch,
)


class StateBackend:
    """Ephemeral in-memory file storage.

    Files are stored in a ``dict[str, str]`` keyed by their canonical path
    (always starting with ``"/"``).  The backend supports the full
    :class:`~homie_core.backend.protocol.BackendProtocol` surface area without
    touching the filesystem.
    """

    def __init__(self) -> None:
        self._files: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(path: str) -> str:
        """Ensure *path* starts with ``"/"`` and has no trailing slash."""
        if not path.startswith("/"):
            path = "/" + path
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        return path

    # ------------------------------------------------------------------
    # ls
    # ------------------------------------------------------------------

    def ls(self, path: str = "/") -> list[FileInfo]:
        """List entries directly under *path*.

        Files are stored with full paths such as ``"/dir/nested.txt"``.  This
        method walks every stored path, finds those that are direct children of
        the requested directory, and returns :class:`FileInfo` objects for each
        unique entry (file or virtual sub-directory).
        """
        dir_path = self._normalise(path)
        # Ensure trailing separator for prefix matching (avoids "/dir" matching "/dirname/…")
        prefix = dir_path if dir_path == "/" else dir_path + "/"

        seen: dict[str, bool] = {}  # name -> is_dir

        for stored_path in self._files:
            if not stored_path.startswith(prefix):
                continue
            remainder = stored_path[len(prefix):]
            if not remainder:
                continue
            parts = remainder.split("/")
            name = parts[0]
            is_dir = len(parts) > 1
            if name not in seen:
                seen[name] = is_dir
            elif is_dir:
                # Promote to directory if we discover deeper entries later
                seen[name] = True

        return [
            FileInfo(
                path=prefix + name if dir_path == "/" else dir_path + "/" + name,
                name=name,
                is_dir=is_dir,
            )
            for name, is_dir in seen.items()
        ]

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        """Read *path* with optional line-based pagination.

        Raises :class:`FileNotFoundError` if *path* does not exist.
        """
        key = self._normalise(path)
        if key not in self._files:
            raise FileNotFoundError(f"No such file: {path!r}")

        lines = self._files[key].splitlines()
        total = len(lines)
        sliced = lines[offset: offset + limit]
        truncated = (offset + limit) < total

        return FileContent(
            content="\n".join(sliced),
            total_lines=total,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, path: str, content: str) -> None:
        """Store *content* under *path*, overwriting any existing entry."""
        self._files[self._normalise(path)] = content

    # ------------------------------------------------------------------
    # edit
    # ------------------------------------------------------------------

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace *old* with *new* inside the stored file at *path*.

        Mirrors :class:`~homie_core.backend.local_filesystem.LocalFilesystemBackend`
        uniqueness logic: if *replace_all* is ``False`` the replacement only
        proceeds when *old* appears exactly once.
        """
        key = self._normalise(path)
        if key not in self._files:
            return EditResult(success=False, occurrences=0, error=f"File not found: {path!r}")

        content = self._files[key]
        count = content.count(old)

        if count == 0:
            return EditResult(success=False, occurrences=0, error=f"String not found in {path!r}")

        if not replace_all and count > 1:
            return EditResult(
                success=False,
                occurrences=count,
                error=(
                    f"String appears {count} times in {path!r}; "
                    "use replace_all=True to replace all occurrences"
                ),
            )

        self._files[key] = content.replace(old, new)
        return EditResult(success=True, occurrences=count)

    # ------------------------------------------------------------------
    # glob
    # ------------------------------------------------------------------

    def glob(self, pattern: str) -> list[str]:
        """Return stored paths matching *pattern* using :mod:`fnmatch`.

        Both the raw stored path (e.g. ``"/file.txt"``) and the path with the
        leading ``"/"`` stripped (e.g. ``"file.txt"``) are tested so callers
        can use either convention.
        """
        results: list[str] = []
        for stored_path in self._files:
            stripped = stored_path.lstrip("/")
            if fnmatch.fnmatch(stored_path, pattern) or fnmatch.fnmatch(stripped, pattern):
                results.append(stored_path)
        return results

    # ------------------------------------------------------------------
    # grep
    # ------------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: Optional[str] = None,
    ) -> list[GrepMatch]:
        """Search for *pattern* (regex) across stored files under *path*.

        *include* is an optional :mod:`fnmatch` pattern applied to the file
        name (e.g. ``"*.py"``).
        """
        search_root = self._normalise(path)
        # Prefix to restrict search scope
        prefix = search_root if search_root == "/" else search_root + "/"

        compiled = re.compile(pattern)
        results: list[GrepMatch] = []

        for stored_path, content in self._files.items():
            # Scope filter
            if search_root != "/" and not stored_path.startswith(prefix):
                continue

            # Include (filename) filter
            if include is not None:
                filename = stored_path.rsplit("/", 1)[-1]
                if not fnmatch.fnmatch(filename, include):
                    continue

            for lineno, line in enumerate(content.splitlines(), start=1):
                if compiled.search(line):
                    results.append(
                        GrepMatch(path=stored_path, line_number=lineno, line=line)
                    )

        return results
