"""Tests for ToolOrchestrator — tool chaining and multi-step execution."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from homie_core.brain.tool_registry import Tool, ToolCall, ToolRegistry, ToolResult
from homie_core.neural.agents.tool_orchestrator import ToolOrchestrator


# ── fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def registry() -> ToolRegistry:
    """A ToolRegistry pre-loaded with two simple tools."""
    reg = ToolRegistry()
    reg.register(Tool(
        name="search",
        description="Search the web",
        execute=lambda query="": f"results for {query}",
    ))
    reg.register(Tool(
        name="summarize",
        description="Summarize text",
        execute=lambda text="": f"summary of [{text}]",
    ))
    reg.register(Tool(
        name="send_email",
        description="Send an email",
        execute=lambda to="", body="": f"email sent to {to}",
    ))
    return reg


@pytest.fixture()
def inference_fn() -> MagicMock:
    return MagicMock(return_value="[]")


@pytest.fixture()
def orchestrator(registry, inference_fn) -> ToolOrchestrator:
    return ToolOrchestrator(tool_registry=registry, inference_fn=inference_fn)


# ── execute_single ──────────────────────────────────────────────────


class TestExecuteSingle:
    def test_success(self, orchestrator):
        result = orchestrator.execute_single("search", {"query": "weather"})
        assert result["success"] is True
        assert "results for weather" in result["output"]
        assert result["error"] is None

    def test_unknown_tool(self, orchestrator):
        result = orchestrator.execute_single("nonexistent_tool", {})
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_tool_with_bad_args(self, registry, inference_fn):
        """Tool exists but called with wrong kwargs — should not crash."""
        registry.register(Tool(
            name="strict",
            description="strict args",
            execute=lambda required_arg: required_arg,
        ))
        orch = ToolOrchestrator(registry, inference_fn)
        result = orch.execute_single("strict", {"wrong_arg": "x"})
        assert result["success"] is False
        assert result["error"] is not None

    def test_tool_exception(self, registry, inference_fn):
        """A tool that raises should return a failure result, not propagate."""
        def _boom(**kw):
            raise RuntimeError("boom")

        registry.register(Tool(name="boom", description="explodes", execute=_boom))
        orch = ToolOrchestrator(registry, inference_fn)
        result = orch.execute_single("boom", {})
        assert result["success"] is False
        assert "boom" in result["error"]


# ── execute_chain ───────────────────────────────────────────────────


class TestExecuteChain:
    def test_simple_chain(self, orchestrator):
        chain = [
            {"tool": "search", "args": {"query": "AI news"}},
            {"tool": "summarize", "args": {"text": "some long article"}},
        ]
        results = orchestrator.execute_chain(chain)
        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is True

    def test_step_reference_resolution(self, orchestrator):
        """$step.0 in step 1's args should resolve to step 0's output."""
        chain = [
            {"tool": "search", "args": {"query": "weather"}},
            {"tool": "summarize", "args": {"text": "$step.0"}, "depends_on": [0]},
        ]
        results = orchestrator.execute_chain(chain)
        assert results[0]["success"] is True
        # Step 1 should have received the output of step 0 as its text arg.
        assert "summary of [results for weather]" in results[1]["output"]

    def test_chain_with_unknown_tool(self, orchestrator):
        chain = [
            {"tool": "nonexistent", "args": {}},
            {"tool": "search", "args": {"query": "test"}},
        ]
        results = orchestrator.execute_chain(chain)
        assert results[0]["success"] is False
        # Chain should continue despite the failure.
        assert results[1]["success"] is True

    def test_empty_chain(self, orchestrator):
        assert orchestrator.execute_chain([]) == []


# ── plan_tool_chain ─────────────────────────────────────────────────


class TestPlanToolChain:
    def test_calls_inference_fn(self, orchestrator, inference_fn):
        inference_fn.return_value = json.dumps([
            {"tool": "search", "args": {"query": "AI"}, "depends_on": []},
        ])
        plan = orchestrator.plan_tool_chain("Find AI news")
        inference_fn.assert_called_once()
        assert len(plan) == 1
        assert plan[0]["tool"] == "search"

    def test_with_explicit_tools(self, orchestrator, inference_fn):
        inference_fn.return_value = "[]"
        orchestrator.plan_tool_chain("goal", available_tools=["search"])
        prompt = inference_fn.call_args[0][0]
        assert "search" in prompt

    def test_handles_markdown_fenced_json(self, orchestrator, inference_fn):
        inference_fn.return_value = '```json\n[{"tool": "search", "args": {}}]\n```'
        plan = orchestrator.plan_tool_chain("test")
        assert len(plan) == 1

    def test_handles_invalid_json(self, orchestrator, inference_fn):
        inference_fn.return_value = "not json at all"
        plan = orchestrator.plan_tool_chain("test")
        assert plan == []

    def test_filters_invalid_steps(self, orchestrator, inference_fn):
        inference_fn.return_value = json.dumps([
            {"tool": "search", "args": {}},
            {"no_tool_key": True},
            "just a string",
        ])
        plan = orchestrator.plan_tool_chain("test")
        assert len(plan) == 1


# ── _resolve_args ───────────────────────────────────────────────────


class TestResolveArgs:
    def test_replaces_step_ref(self):
        results = [{"output": "hello"}]
        resolved = ToolOrchestrator._resolve_args({"x": "$step.0"}, results)
        assert resolved["x"] == "hello"

    def test_leaves_non_string_values_unchanged(self):
        results = [{"output": "hello"}]
        resolved = ToolOrchestrator._resolve_args({"x": 42, "y": True}, results)
        assert resolved == {"x": 42, "y": True}

    def test_out_of_range_ref_left_as_is(self):
        results = []
        resolved = ToolOrchestrator._resolve_args({"x": "$step.5"}, results)
        assert resolved["x"] == "$step.5"
