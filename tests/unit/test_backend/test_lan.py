"""Tests for LANBackend stub."""
from __future__ import annotations

import pytest

from homie_core.backend.lan import LANBackend


def make_backend() -> LANBackend:
    return LANBackend(peer_address="192.168.1.42:8765", auth_key=b"secret")


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_instantiation_works(self):
        backend = make_backend()
        assert backend.peer_address == "192.168.1.42:8765"

    def test_instantiation_stores_peer_address(self):
        backend = LANBackend(peer_address="10.0.0.1:9000", auth_key=b"key")
        assert backend.peer_address == "10.0.0.1:9000"

    def test_instantiation_with_empty_auth_key(self):
        backend = LANBackend(peer_address="localhost", auth_key=b"")
        assert backend is not None


# ---------------------------------------------------------------------------
# _request raises NotImplementedError
# ---------------------------------------------------------------------------

class TestRequest:
    def test_request_raises_not_implemented(self):
        backend = make_backend()
        with pytest.raises(NotImplementedError, match="network"):
            backend._request("read", {"path": "/foo"})

    def test_request_error_message_contains_method(self):
        backend = make_backend()
        with pytest.raises(NotImplementedError, match="ls"):
            backend._request("ls", {"path": "/"})


# ---------------------------------------------------------------------------
# Protocol methods raise NotImplementedError
# ---------------------------------------------------------------------------

class TestProtocolMethods:
    def test_read_raises_not_implemented(self):
        backend = make_backend()
        with pytest.raises(NotImplementedError):
            backend.read("/some/path")

    def test_write_raises_not_implemented(self):
        backend = make_backend()
        with pytest.raises(NotImplementedError):
            backend.write("/some/path", "content")

    def test_edit_raises_not_implemented(self):
        backend = make_backend()
        with pytest.raises(NotImplementedError):
            backend.edit("/some/path", "old", "new")

    def test_ls_raises_not_implemented(self):
        backend = make_backend()
        with pytest.raises(NotImplementedError):
            backend.ls("/")

    def test_glob_raises_not_implemented(self):
        backend = make_backend()
        with pytest.raises(NotImplementedError):
            backend.glob("*.txt")

    def test_grep_raises_not_implemented(self):
        backend = make_backend()
        with pytest.raises(NotImplementedError):
            backend.grep("pattern")

    def test_execute_raises_not_implemented(self):
        backend = make_backend()
        with pytest.raises(NotImplementedError):
            backend.execute("ls -la")
