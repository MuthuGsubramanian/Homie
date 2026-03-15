"""LAN peer backend stub.

All protocol methods delegate to :meth:`_request`, which raises
:class:`NotImplementedError` until the network module is implemented.
"""
from __future__ import annotations

from typing import Optional

from homie_core.backend.protocol import (
    EditResult,
    FileContent,
    FileInfo,
    GrepMatch,
)


class LANBackend:
    """Stub backend that proxies operations to a remote LAN peer.

    Parameters
    ----------
    peer_address:
        Host/IP (and optional port) of the remote Homie peer,
        e.g. ``"192.168.1.42:8765"``.
    auth_key:
        Shared secret used to authenticate requests to the peer.
    """

    def __init__(self, peer_address: str, auth_key: bytes) -> None:
        self.peer_address = peer_address
        self._auth_key = auth_key

    # ------------------------------------------------------------------
    # Internal RPC stub
    # ------------------------------------------------------------------

    def _request(self, method: str, params: dict) -> object:
        """Send an RPC request to the peer.

        Raises
        ------
        NotImplementedError
            Always — this stub requires the network module to be implemented.
        """
        raise NotImplementedError(
            f"LANBackend._request (method={method!r}) requires the network "
            "module (homie_core.network) to be implemented. "
            "Connect to a peer via the network discovery service first."
        )

    # ------------------------------------------------------------------
    # BackendProtocol
    # ------------------------------------------------------------------

    def ls(self, path: str = "/") -> list[FileInfo]:
        return self._request("ls", {"path": path})  # type: ignore[return-value]

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        return self._request("read", {"path": path, "offset": offset, "limit": limit})  # type: ignore[return-value]

    def write(self, path: str, content: str) -> None:
        self._request("write", {"path": path, "content": content})

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self._request(  # type: ignore[return-value]
            "edit",
            {"path": path, "old": old, "new": new, "replace_all": replace_all},
        )

    def glob(self, pattern: str) -> list[str]:
        return self._request("glob", {"pattern": pattern})  # type: ignore[return-value]

    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: Optional[str] = None,
    ) -> list[GrepMatch]:
        return self._request(  # type: ignore[return-value]
            "grep", {"pattern": pattern, "path": path, "include": include}
        )

    def execute(self, command: str, timeout: int = 30) -> object:
        return self._request("execute", {"command": command, "timeout": timeout})
