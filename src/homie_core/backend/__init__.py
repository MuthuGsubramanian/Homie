from __future__ import annotations

from homie_core.backend.composite import CompositeBackend
from homie_core.backend.encrypted import EncryptedVaultBackend
from homie_core.backend.lan import LANBackend
from homie_core.backend.local_filesystem import LocalFilesystemBackend
from homie_core.backend.protocol import (
    BackendProtocol,
    EditResult,
    ExecutableBackend,
    ExecutionResult,
    FileContent,
    FileInfo,
    GrepMatch,
)
from homie_core.backend.state import StateBackend

__all__ = [
    "BackendProtocol",
    "CompositeBackend",
    "EditResult",
    "EncryptedVaultBackend",
    "ExecutableBackend",
    "ExecutionResult",
    "FileContent",
    "FileInfo",
    "GrepMatch",
    "LANBackend",
    "LocalFilesystemBackend",
    "StateBackend",
]
