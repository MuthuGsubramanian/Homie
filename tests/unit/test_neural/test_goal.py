"""Tests for Goal, ThoughtChain, and ThoughtStep data models."""

import json
import time

from homie_core.neural.planning.goal import Goal, ThoughtChain, ThoughtStep


# -- ThoughtStep ----------------------------------------------------------

def test_thought_step_creation():
    step = ThoughtStep(
        id="step-1",
        reasoning="Need to gather data",
        action="query database",
        expected_outcome="list of records",
        agent="research",
        dependencies=[],
    )
    assert step.id == "step-1"
    assert step.status == "pending"
    assert step.result is None
    assert step.dependencies == []


def test_thought_step_to_dict_roundtrip():
    step = ThoughtStep(
        id="s1",
        reasoning="r",
        action="a",
        expected_outcome="o",
        agent="action",
        dependencies=["s0"],
        result={"ok": True},
        status="complete",
    )
    d = step.to_dict()
    restored = ThoughtStep.from_dict(d)
    assert restored.id == step.id
    assert restored.result == {"ok": True}
    assert restored.status == "complete"
    assert restored.dependencies == ["s0"]


# -- ThoughtChain ---------------------------------------------------------

def _make_chain() -> ThoughtChain:
    return ThoughtChain(
        goal="analyze sales data",
        steps=[
            ThoughtStep("s1", "gather", "fetch CSV", "csv loaded", "research", []),
            ThoughtStep("s2", "analyze", "compute stats", "stats dict", "reasoning", ["s1"]),
            ThoughtStep("s3", "report", "write summary", "markdown", "action", ["s2"]),
        ],
    )


def test_chain_creation():
    chain = _make_chain()
    assert chain.status == "thinking"
    assert chain.current_step == 0
    assert len(chain.steps) == 3


def test_chain_to_dict_roundtrip():
    chain = _make_chain()
    d = chain.to_dict()
    restored = ThoughtChain.from_dict(d)
    assert restored.goal == chain.goal
    assert len(restored.steps) == 3
    assert restored.steps[1].dependencies == ["s1"]


def test_chain_json_roundtrip():
    chain = _make_chain()
    raw = chain.to_json()
    restored = ThoughtChain.from_json(raw)
    assert restored.goal == chain.goal
    assert restored.steps[2].agent == "action"


def test_get_ready_steps_initial():
    chain = _make_chain()
    ready = chain.get_ready_steps()
    assert len(ready) == 1
    assert ready[0].id == "s1"


def test_get_ready_steps_after_completion():
    chain = _make_chain()
    chain.steps[0].status = "complete"
    ready = chain.get_ready_steps()
    assert len(ready) == 1
    assert ready[0].id == "s2"


def test_advance():
    chain = _make_chain()
    step = chain.advance()
    assert step is not None
    assert step.id == "s1"
    assert step.status == "active"
    assert chain.status == "executing"


def test_advance_returns_none_when_blocked():
    chain = _make_chain()
    # s1 is active but not complete — s2 depends on s1
    chain.steps[0].status = "active"
    step = chain.advance()
    assert step is None


def test_is_complete():
    chain = _make_chain()
    assert not chain.is_complete
    for s in chain.steps:
        s.status = "complete"
    assert chain.is_complete


def test_has_failed():
    chain = _make_chain()
    assert not chain.has_failed
    chain.steps[1].status = "failed"
    assert chain.has_failed


def test_parallel_ready_steps():
    """Steps with no dependencies can all be ready at once."""
    chain = ThoughtChain(
        goal="parallel work",
        steps=[
            ThoughtStep("a", "r", "act-a", "o", "action", []),
            ThoughtStep("b", "r", "act-b", "o", "research", []),
            ThoughtStep("c", "r", "aggregate", "o", "reasoning", ["a", "b"]),
        ],
    )
    ready = chain.get_ready_steps()
    assert {s.id for s in ready} == {"a", "b"}


# -- Goal -----------------------------------------------------------------

def test_goal_creation():
    goal = Goal(
        id="goal-abc",
        description="prepare quarterly report",
        created_at=time.time(),
    )
    assert goal.priority == 5
    assert goal.lessons_learned == []
    assert goal.parent_id is None
    assert goal.thought_chain is None


def test_goal_to_dict_roundtrip():
    chain = _make_chain()
    goal = Goal(
        id="g1",
        description="do stuff",
        thought_chain=chain,
        priority=3,
        created_at=1000.0,
        lessons_learned=["be faster"],
    )
    d = goal.to_dict()
    restored = Goal.from_dict(d)
    assert restored.id == "g1"
    assert restored.priority == 3
    assert restored.lessons_learned == ["be faster"]
    assert restored.thought_chain is not None
    assert len(restored.thought_chain.steps) == 3


def test_goal_json_roundtrip():
    goal = Goal(
        id="g2",
        description="test json",
        created_at=2000.0,
    )
    raw = goal.to_json()
    restored = Goal.from_json(raw)
    assert restored.id == "g2"
    assert restored.thought_chain is None


def test_goal_new_id():
    id1 = Goal.new_id()
    id2 = Goal.new_id()
    assert id1.startswith("goal-")
    assert id2.startswith("goal-")
    assert id1 != id2
