from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    """Metadata for a single filesystem entry."""
    path: str
    name: str
    is_dir: bool
    size: Optional[int] = None
    modified: Optional[float] = None


@dataclass
class FileContent:
    """Contents of a file read operation, potentially paginated."""
    content: str
    total_lines: int
    truncated: bool = False


@dataclass
class EditResult:
    """Result of an in-place edit operation."""
    success: bool
    occurrences: int = 0
    error: Optional[str] = None


@dataclass
class GrepMatch:
    """A single line match returned by grep."""
    path: str
    line_number: int
    line: str


@dataclass
class ExecutionResult:
    """Result of running a shell command."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class BackendProtocol(Protocol):
    """Structural protocol that all storage backends must satisfy."""

    def ls(self, path: str = "/") -> list[FileInfo]:
        ...

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        ...

    def write(self, path: str, content: str) -> None:
        ...

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> EditResult:
        ...

    def glob(self, pattern: str) -> list[str]:
        ...

    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: Optional[str] = None,
    ) -> list[GrepMatch]:
        ...


@runtime_checkable
class ExecutableBackend(BackendProtocol, Protocol):
    """BackendProtocol extended with shell command execution."""

    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        ...
