"""Source scanner — enumerates files and detects types."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_EXT_TO_TYPE = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
    ".rb": "ruby", ".sh": "bash", ".kt": "kotlin",
    ".md": "markdown", ".txt": "text", ".rst": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".html": "html", ".css": "css", ".sql": "sql",
    ".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".pptx": "pptx",
}

_SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules", ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs"}


@dataclass
class FileInfo:
    path: Path
    file_type: str
    size_bytes: int
    extension: str


class SourceScanner:
    """Scans directories and enumerates files with metadata."""

    def __init__(
        self,
        include_extensions: Optional[set[str]] = None,
        max_depth: int = 10,
        max_file_size_mb: float = 10.0,
    ) -> None:
        self._include = include_extensions
        self._max_depth = max_depth
        self._max_size = int(max_file_size_mb * 1024 * 1024)

    def scan_directory(self, root: Path | str, depth: int = 0) -> list[FileInfo]:
        """Recursively scan a directory for files."""
        root = Path(root)
        results = []

        if depth > self._max_depth:
            return results

        try:
            entries = sorted(root.iterdir())
        except PermissionError:
            return results

        for entry in entries:
            if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
                continue

            if entry.is_dir():
                results.extend(self.scan_directory(entry, depth + 1))
            elif entry.is_file():
                if entry.stat().st_size > self._max_size:
                    continue
                ext = entry.suffix.lower()
                if self._include and ext not in self._include:
                    continue
                file_type = _EXT_TO_TYPE.get(ext, "unknown")
                results.append(FileInfo(
                    path=entry,
                    file_type=file_type,
                    size_bytes=entry.stat().st_size,
                    extension=ext,
                ))

        return results
