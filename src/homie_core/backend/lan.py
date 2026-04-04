"""LAN peer backend — proxies filesystem operations to a remote Homie instance.

Uses HTTP JSON-RPC to communicate with the peer's backend API.  The peer
address and auth key are provided at construction time (typically obtained
from :class:`~homie_core.network.discovery.HomieDiscovery`).
"""
from __future__ import annotations

import json
import logging
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from homie_core.backend.protocol import (
    EditResult,
    FileContent,
    FileInfo,
    GrepMatch,
)

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 30


class LANBackend:
    """Backend that proxies filesystem operations to a remote LAN peer.

    The peer is expected to expose an HTTP JSON-RPC endpoint at
    ``http://<peer_address>/rpc`` that accepts ``{"method": ..., "params": ...}``
    and returns ``{"result": ...}`` or ``{"error": ...}``.

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
        # Normalise base URL
        addr = peer_address.rstrip("/")
        if not addr.startswith("http"):
            addr = f"http://{addr}"
        self._base_url = addr

    # ------------------------------------------------------------------
    # Internal RPC
    # ------------------------------------------------------------------

    def _request(self, method: str, params: dict) -> object:
        """Send a JSON-RPC request to the peer.

        Returns the ``result`` field from the response on success.

        Raises
        ------
        ConnectionError
            If the peer is unreachable.
        RuntimeError
            If the peer returns an application-level error.
        """
        url = f"{self._base_url}/rpc"
        payload = json.dumps({"method": method, "params": params}).encode("utf-8")
        req = Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._auth_key.hex()}")

        try:
            with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                body = json.loads(resp.read())
        except HTTPError as exc:
            try:
                body = json.loads(exc.read())
                msg = body.get("error", str(exc))
            except Exception:
                msg = str(exc)
            raise RuntimeError(
                f"LAN peer {self.peer_address} RPC error ({method}): {msg}"
            ) from exc
        except (URLError, OSError) as exc:
            raise ConnectionError(
                f"LAN peer {self.peer_address} unreachable: {exc}"
            ) from exc

        if "error" in body and body["error"]:
            raise RuntimeError(
                f"LAN peer {self.peer_address} returned error for {method}: {body['error']}"
            )

        return body.get("result")

    # ------------------------------------------------------------------
    # Result deserialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_file_info(raw: dict) -> FileInfo:
        return FileInfo(
            path=raw["path"],
            name=raw["name"],
            is_dir=raw["is_dir"],
            size=raw.get("size"),
            modified=raw.get("modified"),
        )

    @staticmethod
    def _to_file_content(raw: dict) -> FileContent:
        return FileContent(
            content=raw["content"],
            total_lines=raw["total_lines"],
            truncated=raw.get("truncated", False),
        )

    @staticmethod
    def _to_edit_result(raw: dict) -> EditResult:
        return EditResult(
            success=raw["success"],
            occurrences=raw.get("occurrences", 0),
            error=raw.get("error"),
        )

    @staticmethod
    def _to_grep_match(raw: dict) -> GrepMatch:
        return GrepMatch(
            path=raw["path"],
            line_number=raw["line_number"],
            line=raw["line"],
        )

    # ------------------------------------------------------------------
    # BackendProtocol
    # ------------------------------------------------------------------

    def ls(self, path: str = "/") -> list[FileInfo]:
        result = self._request("ls", {"path": path})
        if not isinstance(result, list):
            return []
        return [self._to_file_info(entry) for entry in result]

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        result = self._request("read", {"path": path, "offset": offset, "limit": limit})
        if not isinstance(result, dict):
            return FileContent(content="", total_lines=0, truncated=False)
        return self._to_file_content(result)

    def write(self, path: str, content: str) -> None:
        self._request("write", {"path": path, "content": content})

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> EditResult:
        result = self._request(
            "edit",
            {"path": path, "old": old, "new": new, "replace_all": replace_all},
        )
        if not isinstance(result, dict):
            return EditResult(success=False, error="Unexpected response from peer")
        return self._to_edit_result(result)

    def glob(self, pattern: str) -> list[str]:
        result = self._request("glob", {"pattern": pattern})
        if not isinstance(result, list):
            return []
        return result

    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: Optional[str] = None,
    ) -> list[GrepMatch]:
        result = self._request(
            "grep", {"pattern": pattern, "path": path, "include": include}
        )
        if not isinstance(result, list):
            return []
        return [self._to_grep_match(entry) for entry in result]

    def execute(self, command: str, timeout: int = 30) -> object:
        return self._request("execute", {"command": command, "timeout": timeout})
