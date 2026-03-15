from __future__ import annotations

import pytest

from homie_core.middleware.subagent import SubAgentMiddleware


def make_mw(factory=None) -> SubAgentMiddleware:
    if factory is None:
        factory = lambda desc: f"result for: {desc}"
    return SubAgentMiddleware(agent_factory=factory)


# ---------------------------------------------------------------------------
# modify_tools
# ---------------------------------------------------------------------------

def test_modify_tools_adds_task_tool():
    mw = make_mw()
    tools = mw.modify_tools([])
    names = [t["name"] for t in tools]
    assert "task" in names


def test_modify_tools_preserves_existing_tools():
    mw = make_mw()
    existing = [{"name": "search"}, {"name": "write_todos"}]
    tools = mw.modify_tools(existing)
    names = [t["name"] for t in tools]
    assert "search" in names
    assert "write_todos" in names
    assert "task" in names
    assert len(tools) == 3


def test_task_tool_has_description():
    mw = make_mw()
    tools = mw.modify_tools([])
    task_tool = next(t for t in tools if t["name"] == "task")
    assert "description" in task_tool
    assert len(task_tool["description"]) > 0


# ---------------------------------------------------------------------------
# wrap_tool_call — task tool
# ---------------------------------------------------------------------------

def test_wrap_tool_call_task_calls_agent_factory():
    calls = []

    def factory(desc: str) -> str:
        calls.append(desc)
        return "agent response"

    mw = SubAgentMiddleware(agent_factory=factory)
    mw.wrap_tool_call("task", {"description": "Do something complex"})
    assert calls == ["Do something complex"]


def test_wrap_tool_call_task_stores_result_in_args():
    mw = make_mw(factory=lambda d: "the answer")
    args = {"description": "What is 2+2?"}
    result = mw.wrap_tool_call("task", args)
    assert result["_subagent_result"] == "the answer"


def test_wrap_tool_call_task_passes_description_to_factory():
    received = {}

    def factory(desc: str) -> str:
        received["desc"] = desc
        return "ok"

    mw = SubAgentMiddleware(agent_factory=factory)
    mw.wrap_tool_call("task", {"description": "Analyze this file"})
    assert received["desc"] == "Analyze this file"


def test_wrap_tool_call_task_factory_exception_stores_error_message():
    def bad_factory(desc: str) -> str:
        raise RuntimeError("agent blew up")

    mw = SubAgentMiddleware(agent_factory=bad_factory)
    result = mw.wrap_tool_call("task", {"description": "dangerous task"})
    assert "_subagent_result" in result
    assert "Sub-agent error" in result["_subagent_result"]
    assert "agent blew up" in result["_subagent_result"]


def test_wrap_tool_call_task_empty_description_passthrough():
    calls = []

    def factory(desc: str) -> str:
        calls.append(desc)
        return "result"

    mw = SubAgentMiddleware(agent_factory=factory)
    result = mw.wrap_tool_call("task", {"description": ""})
    # Empty description — factory should NOT be called
    assert calls == []
    assert "_subagent_result" not in result


def test_wrap_tool_call_task_missing_description_passthrough():
    calls = []

    def factory(desc: str) -> str:
        calls.append(desc)
        return "result"

    mw = SubAgentMiddleware(agent_factory=factory)
    result = mw.wrap_tool_call("task", {})
    assert calls == []
    assert "_subagent_result" not in result


# ---------------------------------------------------------------------------
# wrap_tool_call — other tools (passthrough)
# ---------------------------------------------------------------------------

def test_wrap_tool_call_other_tool_passthrough():
    calls = []
    mw = make_mw(factory=lambda d: calls.append(d) or "x")
    args = {"query": "foo", "limit": 5}
    result = mw.wrap_tool_call("search", args)
    assert result == args
    assert calls == []


def test_wrap_tool_call_other_tool_returns_args_unchanged():
    mw = make_mw()
    args = {"path": "/tmp/file.txt", "content": "hello"}
    result = mw.wrap_tool_call("write_file", args)
    assert result is args


def test_wrap_tool_call_other_tool_does_not_add_subagent_result():
    mw = make_mw()
    result = mw.wrap_tool_call("read_file", {"path": "/tmp/x"})
    assert "_subagent_result" not in result


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_name_and_order():
    mw = make_mw()
    assert mw.name == "subagent"
    assert mw.order == 50
