from __future__ import annotations

import pytest
from homie_core.backend.protocol import (
    BackendProtocol,
    EditResult,
    ExecutableBackend,
    ExecutionResult,
    FileContent,
    FileInfo,
    GrepMatch,
)


# ---------------------------------------------------------------------------
# FileInfo
# ---------------------------------------------------------------------------

class TestFileInfo:
    def test_required_fields(self):
        fi = FileInfo(path="/foo/bar.txt", name="bar.txt", is_dir=False)
        assert fi.path == "/foo/bar.txt"
        assert fi.name == "bar.txt"
        assert fi.is_dir is False

    def test_optional_defaults(self):
        fi = FileInfo(path="/a", name="a", is_dir=True)
        assert fi.size is None
        assert fi.modified is None

    def test_optional_fields_set(self):
        fi = FileInfo(path="/a", name="a", is_dir=False, size=1024, modified=1700000000.0)
        assert fi.size == 1024
        assert fi.modified == 1700000000.0

    def test_is_dir_true(self):
        fi = FileInfo(path="/some/dir", name="dir", is_dir=True)
        assert fi.is_dir is True


# ---------------------------------------------------------------------------
# FileContent
# ---------------------------------------------------------------------------

class TestFileContent:
    def test_required_fields(self):
        fc = FileContent(content="hello\nworld", total_lines=2)
        assert fc.content == "hello\nworld"
        assert fc.total_lines == 2

    def test_truncated_default_false(self):
        fc = FileContent(content="x", total_lines=1)
        assert fc.truncated is False

    def test_truncated_set_true(self):
        fc = FileContent(content="partial", total_lines=500, truncated=True)
        assert fc.truncated is True


# ---------------------------------------------------------------------------
# EditResult
# ---------------------------------------------------------------------------

class TestEditResult:
    def test_success_no_error(self):
        er = EditResult(success=True)
        assert er.success is True
        assert er.occurrences == 0
        assert er.error is None

    def test_failure_with_error(self):
        er = EditResult(success=False, error="not found")
        assert er.success is False
        assert er.error == "not found"

    def test_occurrences_field(self):
        er = EditResult(success=False, occurrences=3, error="not unique")
        assert er.occurrences == 3


# ---------------------------------------------------------------------------
# GrepMatch
# ---------------------------------------------------------------------------

class TestGrepMatch:
    def test_fields(self):
        gm = GrepMatch(path="/a/b.py", line_number=42, line="    result = foo()")
        assert gm.path == "/a/b.py"
        assert gm.line_number == 42
        assert gm.line == "    result = foo()"


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

class TestExecutionResult:
    def test_required_fields(self):
        er = ExecutionResult(stdout="hello\n", stderr="", exit_code=0)
        assert er.stdout == "hello\n"
        assert er.stderr == ""
        assert er.exit_code == 0

    def test_timed_out_default_false(self):
        er = ExecutionResult(stdout="", stderr="", exit_code=1)
        assert er.timed_out is False

    def test_timed_out_set_true(self):
        er = ExecutionResult(stdout="", stderr="killed", exit_code=-1, timed_out=True)
        assert er.timed_out is True


# ---------------------------------------------------------------------------
# Protocol structural checks
# ---------------------------------------------------------------------------

class TestBackendProtocol:
    def test_is_protocol(self):
        # BackendProtocol should be a Protocol class (runtime_checkable)
        from typing import runtime_checkable
        # runtime_checkable protocols have __protocol_attrs__
        assert hasattr(BackendProtocol, "__protocol_attrs__") or hasattr(BackendProtocol, "_is_protocol")

    def test_backend_protocol_methods_exist(self):
        for method_name in ("ls", "read", "write", "edit", "glob", "grep"):
            assert hasattr(BackendProtocol, method_name), f"Missing method: {method_name}"

    def test_executable_backend_adds_execute(self):
        assert hasattr(ExecutableBackend, "execute")

    def test_concrete_class_passes_isinstance_check(self):
        class FakeBackend:
            def ls(self, path="/"): ...
            def read(self, path, offset=0, limit=100): ...
            def write(self, path, content): ...
            def edit(self, path, old, new, replace_all=False): ...
            def glob(self, pattern): ...
            def grep(self, pattern, path="/", include=None): ...

        fb = FakeBackend()
        assert isinstance(fb, BackendProtocol)

    def test_concrete_class_missing_method_fails_isinstance(self):
        class IncompleteBackend:
            def ls(self, path="/"): ...
            # missing read, write, edit, glob, grep

        ib = IncompleteBackend()
        assert not isinstance(ib, BackendProtocol)

    def test_executable_backend_isinstance_check(self):
        class FakeExecBackend:
            def ls(self, path="/"): ...
            def read(self, path, offset=0, limit=100): ...
            def write(self, path, content): ...
            def edit(self, path, old, new, replace_all=False): ...
            def glob(self, pattern): ...
            def grep(self, pattern, path="/", include=None): ...
            def execute(self, command, timeout=30): ...

        feb = FakeExecBackend()
        assert isinstance(feb, ExecutableBackend)
