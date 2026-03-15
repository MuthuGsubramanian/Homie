from __future__ import annotations

import pytest

from homie_core.middleware.hitl import HITLMiddleware


def make_mw(**kwargs) -> HITLMiddleware:
    return HITLMiddleware(**kwargs)


# ---------------------------------------------------------------------------
# Non-gated tool passes through
# ---------------------------------------------------------------------------

def test_non_gated_tool_passes_through():
    mw = make_mw()
    args = {"query": "search term"}
    result = mw.wrap_tool_call("search", args)
    assert result == args


def test_read_file_passes_through():
    mw = make_mw()
    args = {"path": "/foo.txt"}
    result = mw.wrap_tool_call("read_file", args)
    assert result == args


def test_list_dir_passes_through():
    mw = make_mw()
    args = {"path": "/"}
    result = mw.wrap_tool_call("list_dir", args)
    assert result == args


# ---------------------------------------------------------------------------
# Gated tool is blocked with _hitl_pending in args
# ---------------------------------------------------------------------------

def test_gated_run_command_blocked():
    mw = make_mw()
    args = {"command": "rm -rf /"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


def test_gated_write_file_blocked():
    mw = make_mw()
    args = {"path": "/etc/passwd", "content": "pwned"}
    result = mw.wrap_tool_call("write_file", args)
    assert result is None


def test_gated_run_command_sets_hitl_pending():
    mw = make_mw()
    args = {"command": "git push --force"}
    mw.wrap_tool_call("run_command", args)
    assert "_hitl_pending" in args


def test_hitl_pending_contains_tool_name():
    mw = make_mw()
    args = {"command": "git push"}
    mw.wrap_tool_call("run_command", args)
    assert args["_hitl_pending"]["tool"] == "run_command"


def test_hitl_pending_contains_args():
    mw = make_mw()
    args = {"command": "git push", "extra": 42}
    mw.wrap_tool_call("run_command", args)
    pending = args["_hitl_pending"]
    assert "args" in pending
    assert pending["args"]["command"] == "git push"


def test_hitl_pending_contains_options():
    mw = make_mw()
    args = {"command": "echo hi"}
    mw.wrap_tool_call("run_command", args)
    pending = args["_hitl_pending"]
    assert "options" in pending
    assert "approve" in pending["options"]
    assert "reject" in pending["options"]
    assert "auto_approve" in pending["options"]


# ---------------------------------------------------------------------------
# After auto_approve, gated tool passes through
# ---------------------------------------------------------------------------

def test_after_auto_approve_run_command_passes():
    mw = make_mw()
    mw.approve("run_command", auto=True)
    args = {"command": "git status"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is not None
    assert result["command"] == "git status"


def test_auto_approve_only_affects_named_tool():
    mw = make_mw()
    mw.approve("run_command", auto=True)
    # write_file should still be gated
    args = {"path": "/tmp/x", "content": "y"}
    result = mw.wrap_tool_call("write_file", args)
    assert result is None


def test_is_auto_approved_returns_true_after_auto_approve():
    mw = make_mw()
    mw.approve("run_command", auto=True)
    assert mw.is_auto_approved("run_command") is True


def test_is_auto_approved_returns_false_before_approve():
    mw = make_mw()
    assert mw.is_auto_approved("run_command") is False


# ---------------------------------------------------------------------------
# approve with auto=False doesn't add to auto_approved
# ---------------------------------------------------------------------------

def test_approve_without_auto_does_not_add_to_auto_approved():
    mw = make_mw()
    mw.approve("run_command", auto=False)
    assert mw.is_auto_approved("run_command") is False


def test_approve_without_auto_tool_still_blocked_next_call():
    mw = make_mw()
    mw.approve("run_command", auto=False)
    # Next call should still be blocked (one-shot approval not stored)
    args = {"command": "git status"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


# ---------------------------------------------------------------------------
# Custom gated_tools set works
# ---------------------------------------------------------------------------

def test_custom_gated_tools_only_gates_those():
    mw = make_mw(gated_tools={"dangerous_op"})
    # run_command should NOT be gated now
    args = {"command": "rm -rf /"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is not None


def test_custom_gated_tools_gates_custom_tool():
    mw = make_mw(gated_tools={"dangerous_op"})
    args = {"op": "nuke"}
    result = mw.wrap_tool_call("dangerous_op", args)
    assert result is None


def test_custom_gated_tools_hitl_pending_has_correct_tool():
    mw = make_mw(gated_tools={"dangerous_op"})
    args = {"op": "nuke"}
    mw.wrap_tool_call("dangerous_op", args)
    assert args["_hitl_pending"]["tool"] == "dangerous_op"


# ---------------------------------------------------------------------------
# reject() is a no-op (call was already blocked)
# ---------------------------------------------------------------------------

def test_reject_does_not_raise():
    mw = make_mw()
    mw.reject()  # should not raise


def test_reject_does_not_affect_auto_approved():
    mw = make_mw()
    mw.approve("run_command", auto=True)
    mw.reject()
    assert mw.is_auto_approved("run_command") is True


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_name_and_order():
    mw = make_mw()
    assert mw.name == "hitl"
    assert mw.order == 3
