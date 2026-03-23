"""Unit tests for the MetaAgent orchestrator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homie_core.neural.communication.agent_bus import AgentBus, AgentMessage
from homie_core.neural.config import NeuralConfig
from homie_core.neural.meta_agent import MetaAgent
from homie_core.neural.planning.goal import Goal, ThoughtChain, ThoughtStep
from homie_core.neural.planning.goal_memory import GoalMemory
from homie_core.neural.planning.planner import Planner
from homie_core.neural.planning.replanner import Replanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inference_fn(responses: list[str] | None = None):
    """Return an inference_fn that cycles through *responses*."""
    if responses is None:
        responses = ['"moderate"']
    idx = {"i": 0}

    def infer(prompt: str, **kw) -> str:
        r = responses[idx["i"] % len(responses)]
        idx["i"] += 1
        return r

    return infer


def _simple_plan_json(n_steps: int = 2, agents: list[str] | None = None) -> str:
    agents = agents or ["reasoning", "action"]
    steps = []
    deps: list[str] = []
    for i in range(n_steps):
        sid = f"step-{i + 1}"
        steps.append({
            "id": sid,
            "reasoning": f"Reason for step {i + 1}",
            "action": f"Do thing {i + 1}",
            "expected_outcome": f"Outcome {i + 1}",
            "agent": agents[i % len(agents)],
            "dependencies": deps.copy(),
        })
        deps = [sid]
    return json.dumps({"steps": steps})


def _make_agent(name: str, result: dict | None = None):
    """Create a mock agent with an async process method."""
    agent = MagicMock()
    agent.name = name

    async def mock_process(message: AgentMessage) -> AgentMessage:
        return AgentMessage(
            from_agent=name,
            to_agent=message.from_agent,
            message_type="result",
            content=result or {"status": "success", "output": f"{name} done"},
            parent_goal_id=message.parent_goal_id,
        )

    agent.process = mock_process
    agent.validate = MagicMock(return_value={
        "valid": True,
        "score": 0.95,
        "issues": [],
        "suggestions": [],
    })
    return agent


def _build_meta(
    inference_responses: list[str] | None = None,
    config: NeuralConfig | None = None,
    agents: dict | None = None,
) -> MetaAgent:
    """Convenience factory that wires up a MetaAgent with mocks."""
    if inference_responses is None:
        inference_responses = ["moderate", _simple_plan_json()]

    inference_fn = _make_inference_fn(inference_responses)
    bus = AgentBus()

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        goal_memory = GoalMemory(db_path)
        planner = Planner(inference_fn)
        replanner = Replanner(inference_fn)
        default_agents = agents or {
            "reasoning": _make_agent("reasoning"),
            "action": _make_agent("action"),
            "validation": _make_agent("validation"),
            "research": _make_agent("research"),
        }
        meta = MetaAgent(
            inference_fn=inference_fn,
            goal_memory=goal_memory,
            planner=planner,
            replanner=replanner,
            agents=default_agents,
            agent_bus=bus,
            config=config or NeuralConfig(),
        )
        # Keep references alive for test duration
        meta._test_bus = bus
        meta._test_db_path = db_path
        return meta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture()
def meta_agent(tmp_db):
    plan = _simple_plan_json()
    inference_fn = _make_inference_fn(["moderate", plan])
    bus = AgentBus()
    goal_memory = GoalMemory(tmp_db)
    planner = Planner(inference_fn)
    replanner = Replanner(inference_fn, max_replans=2)
    agents = {
        "reasoning": _make_agent("reasoning"),
        "action": _make_agent("action"),
        "validation": _make_agent("validation"),
        "research": _make_agent("research"),
    }
    meta = MetaAgent(
        inference_fn=inference_fn,
        goal_memory=goal_memory,
        planner=planner,
        replanner=replanner,
        agents=agents,
        agent_bus=bus,
        config=NeuralConfig(),
    )
    yield meta
    bus.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubmitGoal:
    def test_submit_goal_creates_and_plans(self, meta_agent: MetaAgent):
        goal = meta_agent.submit_goal("Summarise last month's expenses")

        assert goal.id.startswith("goal-")
        assert goal.description == "Summarise last month's expenses"
        assert goal.thought_chain is not None
        assert len(goal.thought_chain.steps) >= 1
        assert goal.id in meta_agent._active_goals

    def test_submit_goal_persists_to_memory(self, meta_agent: MetaAgent):
        goal = meta_agent.submit_goal("Do something useful")
        retrieved = meta_agent.goal_memory.get_goal(goal.id)
        assert retrieved is not None
        assert retrieved.description == goal.description

    def test_submit_goal_respects_max_concurrent(self, tmp_db):
        plan = _simple_plan_json()
        inference_fn = _make_inference_fn(["simple", plan] * 10)
        bus = AgentBus()
        goal_memory = GoalMemory(tmp_db)
        config = NeuralConfig(max_concurrent_goals=2)
        meta = MetaAgent(
            inference_fn=inference_fn,
            goal_memory=goal_memory,
            planner=Planner(inference_fn),
            replanner=Replanner(inference_fn),
            agents={"reasoning": _make_agent("reasoning")},
            agent_bus=bus,
            config=config,
        )
        meta.submit_goal("Goal 1")
        meta.submit_goal("Goal 2")
        with pytest.raises(RuntimeError, match="Max concurrent goals"):
            meta.submit_goal("Goal 3")
        bus.shutdown()


class TestExecuteGoal:
    def test_execute_goal_delegates_to_agents(self, meta_agent: MetaAgent):
        goal = meta_agent.submit_goal("Analyse sales data")
        result = meta_agent.execute_goal(goal)

        assert result["status"] == "completed"
        assert result["goal_id"] == goal.id
        # All steps should be complete
        for step in goal.thought_chain.steps:
            assert step.status == "complete"

    def test_execute_goal_returns_validation(self, meta_agent: MetaAgent):
        goal = meta_agent.submit_goal("Check compliance")
        result = meta_agent.execute_goal(goal)

        assert result["validation"] is not None
        assert result["validation"]["valid"] is True

    def test_execute_goal_no_thought_chain(self, meta_agent: MetaAgent):
        goal = Goal(id=Goal.new_id(), description="Orphan goal")
        result = meta_agent.execute_goal(goal)
        assert result["status"] == "failed"
        assert "no thought chain" in result.get("error", "").lower()

    def test_execute_goal_handles_failure_with_replan(self, tmp_db):
        """When a step fails, the replanner should kick in."""
        # Inference: classify → plan → (replan produces new steps)
        replan_json = json.dumps({
            "steps": [{
                "id": "step-alt-1",
                "reasoning": "Alternative approach",
                "action": "Try plan B",
                "expected_outcome": "Success via B",
                "agent": "reasoning",
                "dependencies": [],
            }]
        })
        responses = ["simple", _simple_plan_json(1, ["action"]), replan_json]
        inference_fn = _make_inference_fn(responses)
        bus = AgentBus()

        # Create a failing action agent
        fail_agent = MagicMock()
        fail_agent.name = "action"
        call_count = {"n": 0}

        async def fail_process(msg):
            call_count["n"] += 1
            raise RuntimeError("Simulated failure")

        fail_agent.process = fail_process

        meta = MetaAgent(
            inference_fn=inference_fn,
            goal_memory=GoalMemory(tmp_db),
            planner=Planner(inference_fn),
            replanner=Replanner(inference_fn, max_replans=2),
            agents={
                "action": fail_agent,
                "reasoning": _make_agent("reasoning"),
                "validation": _make_agent("validation"),
            },
            agent_bus=bus,
        )

        goal = meta.submit_goal("Risky operation")
        result = meta.execute_goal(goal)

        # Should have attempted replan (action fails, then reasoning succeeds)
        assert result["goal_id"] == goal.id
        bus.shutdown()


class TestCancelGoal:
    def test_cancel_active_goal(self, meta_agent: MetaAgent):
        goal = meta_agent.submit_goal("Long running task")
        assert meta_agent.cancel_goal(goal.id) is True
        assert goal.id not in meta_agent._active_goals
        assert goal.outcome == "cancelled"

    def test_cancel_nonexistent_goal(self, meta_agent: MetaAgent):
        assert meta_agent.cancel_goal("goal-does-not-exist") is False


class TestGetActiveGoals:
    def test_get_active_goals(self, meta_agent: MetaAgent):
        g1 = meta_agent.submit_goal("Goal A")
        g2 = meta_agent.submit_goal("Goal B")
        active = meta_agent.get_active_goals()
        ids = [g.id for g in active]
        assert g1.id in ids
        assert g2.id in ids

    def test_get_active_goals_sorted_by_priority(self, tmp_db):
        plan = _simple_plan_json()
        inference_fn = _make_inference_fn(["simple", plan] * 10)
        bus = AgentBus()
        meta = MetaAgent(
            inference_fn=inference_fn,
            goal_memory=GoalMemory(tmp_db),
            planner=Planner(inference_fn),
            replanner=Replanner(inference_fn),
            agents={"reasoning": _make_agent("reasoning")},
            agent_bus=bus,
            config=NeuralConfig(max_concurrent_goals=10),
        )
        meta.submit_goal("Low priority", priority=10)
        meta.submit_goal("High priority", priority=1)
        active = meta.get_active_goals()
        assert active[0].priority <= active[1].priority
        bus.shutdown()


class TestGetGoalStatus:
    def test_status_of_active_goal(self, meta_agent: MetaAgent):
        goal = meta_agent.submit_goal("Check status")
        status = meta_agent.get_goal_status(goal.id)
        assert status["goal_id"] == goal.id
        assert "status" in status

    def test_status_of_unknown_goal(self, meta_agent: MetaAgent):
        status = meta_agent.get_goal_status("goal-nonexistent")
        assert status["status"] == "not_found"


class TestValidationRunsAfterCompletion:
    def test_validation_called(self, meta_agent: MetaAgent):
        goal = meta_agent.submit_goal("Validate me")
        result = meta_agent.execute_goal(goal)
        assert result["status"] == "completed"
        assert result["validation"] is not None
        assert result["validation"]["valid"] is True
        meta_agent.agents["validation"].validate.assert_called_once()


class TestSupervisedMode:
    def test_supervised_mode_pauses_before_action(self, tmp_db):
        """In supervised mode, execution pauses before action agent steps."""
        plan = _simple_plan_json(2, ["reasoning", "action"])
        inference_fn = _make_inference_fn(["simple", plan])
        bus = AgentBus()
        config = NeuralConfig(autonomy_level="supervised")

        meta = MetaAgent(
            inference_fn=inference_fn,
            goal_memory=GoalMemory(tmp_db),
            planner=Planner(inference_fn),
            replanner=Replanner(inference_fn),
            agents={
                "reasoning": _make_agent("reasoning"),
                "action": _make_agent("action"),
                "validation": _make_agent("validation"),
            },
            agent_bus=bus,
            config=config,
        )

        goal = meta.submit_goal("Supervised task")
        result = meta.execute_goal(goal)

        # Should pause before the action step
        assert result["status"] == "paused"
        assert result["paused_step"] is not None
        bus.shutdown()


class TestNeuralConfig:
    def test_default_config(self):
        cfg = NeuralConfig()
        assert cfg.enabled is True
        assert cfg.autonomy_level == "full"
        assert cfg.max_concurrent_goals == 5

    def test_invalid_autonomy_level(self):
        with pytest.raises(ValueError, match="autonomy_level"):
            NeuralConfig(autonomy_level="yolo")

    def test_custom_config(self):
        cfg = NeuralConfig(
            autonomy_level="assisted",
            max_concurrent_goals=10,
            validation_threshold=0.9,
        )
        assert cfg.autonomy_level == "assisted"
        assert cfg.max_concurrent_goals == 10
        assert cfg.validation_threshold == 0.9
