"""Tests for WorkflowEngine — conditional workflows with error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from homie_core.brain.tool_registry import Tool, ToolRegistry
from homie_core.neural.agents.tool_orchestrator import ToolOrchestrator
from homie_core.neural.agents.workflow_engine import WorkflowEngine, WorkflowStep


# ── fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(name="fetch", description="Fetch data", execute=lambda url="": f"data from {url}"))
    reg.register(Tool(name="transform", description="Transform data", execute=lambda data="": f"transformed({data})"))
    reg.register(Tool(name="save", description="Save data", execute=lambda data="": "saved"))
    return reg


@pytest.fixture()
def failing_registry() -> ToolRegistry:
    """Registry with a tool that always fails."""
    reg = ToolRegistry()
    reg.register(Tool(name="good", description="works", execute=lambda: "ok"))

    def _fail(**kw):
        raise RuntimeError("intentional failure")

    reg.register(Tool(name="bad", description="fails", execute=_fail))
    return reg


@pytest.fixture()
def inference_fn() -> MagicMock:
    return MagicMock(return_value="[]")


@pytest.fixture()
def engine(registry, inference_fn) -> WorkflowEngine:
    orch = ToolOrchestrator(tool_registry=registry, inference_fn=inference_fn)
    return WorkflowEngine(orch)


@pytest.fixture()
def failing_engine(failing_registry, inference_fn) -> WorkflowEngine:
    orch = ToolOrchestrator(tool_registry=failing_registry, inference_fn=inference_fn)
    return WorkflowEngine(orch)


# ── run_workflow ────────────────────────────────────────────────────


class TestRunWorkflow:
    def test_all_steps_succeed(self, engine):
        steps = [
            WorkflowStep(tool="fetch", args={"url": "http://example.com"}),
            WorkflowStep(tool="transform", args={"data": "raw"}),
            WorkflowStep(tool="save", args={"data": "clean"}),
        ]
        result = engine.run_workflow(steps)
        assert result["status"] == "completed"
        assert result["completed"] == 3
        assert result["failed"] == 0
        assert result["skipped"] == 0

    def test_stop_on_failure(self, failing_engine):
        steps = [
            WorkflowStep(tool="good"),
            WorkflowStep(tool="bad", on_failure="stop"),
            WorkflowStep(tool="good"),
        ]
        result = failing_engine.run_workflow(steps)
        assert result["status"] == "partial"
        assert result["completed"] == 1
        assert result["failed"] == 1
        assert result["skipped"] == 1

    def test_skip_on_failure(self, failing_engine):
        steps = [
            WorkflowStep(tool="bad", on_failure="skip"),
            WorkflowStep(tool="good"),
        ]
        result = failing_engine.run_workflow(steps)
        assert result["status"] == "partial"
        assert result["completed"] == 1
        assert result["failed"] == 1

    def test_retry_on_failure(self, failing_registry, inference_fn):
        """With on_failure='retry', the engine should attempt max_retries+1 times."""
        call_count = 0

        def _fail_then_succeed(**kw):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient")
            return "recovered"

        failing_registry.register(Tool(name="flaky", description="flaky", execute=_fail_then_succeed))
        orch = ToolOrchestrator(failing_registry, inference_fn)
        eng = WorkflowEngine(orch)

        steps = [WorkflowStep(tool="flaky", on_failure="retry", max_retries=2)]
        result = eng.run_workflow(steps)
        assert result["status"] == "completed"
        assert call_count == 2

    def test_condition_true_runs_step(self, engine):
        steps = [
            WorkflowStep(tool="fetch", args={"url": "x"}),
            WorkflowStep(tool="save", args={"data": "y"}, condition="True"),
        ]
        result = engine.run_workflow(steps)
        assert result["completed"] == 2
        assert result["skipped"] == 0

    def test_condition_false_skips_step(self, engine):
        steps = [
            WorkflowStep(tool="fetch", args={"url": "x"}),
            WorkflowStep(tool="save", args={"data": "y"}, condition="False"),
        ]
        result = engine.run_workflow(steps)
        assert result["completed"] == 1
        assert result["skipped"] == 1

    def test_condition_references_last_result(self, engine):
        steps = [
            WorkflowStep(tool="fetch", args={"url": "x"}),
            WorkflowStep(tool="save", condition="last.get('success') == True"),
        ]
        result = engine.run_workflow(steps)
        assert result["completed"] == 2

    def test_empty_workflow(self, engine):
        result = engine.run_workflow([])
        assert result["status"] == "completed"
        assert result["completed"] == 0


# ── create_workflow_from_goal ───────────────────────────────────────


class TestCreateWorkflowFromGoal:
    def test_parses_llm_response(self, engine, inference_fn):
        inference_fn.return_value = json.dumps([
            {"tool": "fetch", "args": {"url": "http://a.com"}, "on_failure": "stop"},
            {"tool": "transform", "args": {"data": "x"}, "condition": None, "on_failure": "skip"},
        ])
        steps = engine.create_workflow_from_goal("Fetch and transform data")
        assert len(steps) == 2
        assert steps[0].tool == "fetch"
        assert steps[1].on_failure == "skip"

    def test_handles_bad_llm_response(self, engine, inference_fn):
        inference_fn.return_value = "not valid json"
        steps = engine.create_workflow_from_goal("broken")
        assert steps == []


# ── _evaluate_condition ─────────────────────────────────────────────


class TestEvaluateCondition:
    def test_empty_condition_is_true(self):
        assert WorkflowEngine._evaluate_condition("", []) is True

    def test_invalid_condition_is_false(self):
        assert WorkflowEngine._evaluate_condition("undefined_var", []) is False

    def test_results_accessible(self):
        results = [{"success": True, "output": "hello"}]
        assert WorkflowEngine._evaluate_condition("len(results) == 1", results) is True

    def test_last_accessible(self):
        results = [{"success": True}]
        assert WorkflowEngine._evaluate_condition("last.get('success')", results) is True
