"""Tests for GoalMemory — SQLite persistence of Goal objects."""

import time

import pytest

from homie_core.neural.planning.goal import Goal, ThoughtChain, ThoughtStep
from homie_core.neural.planning.goal_memory import GoalMemory


@pytest.fixture
def memory(tmp_path):
    db = tmp_path / "test_learning.db"
    return GoalMemory(db_path=db)


def _make_goal(goal_id: str = "g1", desc: str = "test goal", **kwargs) -> Goal:
    return Goal(
        id=goal_id,
        description=desc,
        created_at=kwargs.get("created_at", time.time()),
        priority=kwargs.get("priority", 5),
        parent_id=kwargs.get("parent_id"),
        thought_chain=kwargs.get("thought_chain"),
        completed_at=kwargs.get("completed_at"),
        outcome=kwargs.get("outcome"),
        lessons_learned=kwargs.get("lessons_learned", []),
    )


def _make_chain() -> ThoughtChain:
    return ThoughtChain(
        goal="do something",
        steps=[
            ThoughtStep("s1", "reason", "act", "result", "action", []),
        ],
    )


# -- save and get ---------------------------------------------------------

def test_save_and_get(memory):
    goal = _make_goal("g1", "my goal")
    memory.save_goal(goal)
    loaded = memory.get_goal("g1")
    assert loaded is not None
    assert loaded.id == "g1"
    assert loaded.description == "my goal"


def test_get_nonexistent(memory):
    assert memory.get_goal("nope") is None


def test_save_with_thought_chain(memory):
    chain = _make_chain()
    goal = _make_goal("g2", "with chain", thought_chain=chain)
    memory.save_goal(goal)
    loaded = memory.get_goal("g2")
    assert loaded.thought_chain is not None
    assert loaded.thought_chain.goal == "do something"
    assert len(loaded.thought_chain.steps) == 1


def test_save_with_lessons(memory):
    goal = _make_goal("g3", lessons_learned=["faster next time", "use cache"])
    memory.save_goal(goal)
    loaded = memory.get_goal("g3")
    assert loaded.lessons_learned == ["faster next time", "use cache"]


def test_upsert(memory):
    goal = _make_goal("g1", "original")
    memory.save_goal(goal)
    goal.description = "updated"
    goal.outcome = "success"
    goal.completed_at = time.time()
    memory.save_goal(goal)
    loaded = memory.get_goal("g1")
    assert loaded.description == "updated"
    assert loaded.outcome == "success"


# -- list_active / list_completed ----------------------------------------

def test_list_active(memory):
    for i in range(3):
        memory.save_goal(_make_goal(f"g{i}", f"goal {i}", priority=i + 1))
    active = memory.list_active()
    assert len(active) == 3
    # Should be ordered by priority ASC
    assert [g.priority for g in active] == [1, 2, 3]


def test_list_completed(memory):
    now = time.time()
    for i in range(5):
        memory.save_goal(
            _make_goal(
                f"g{i}",
                f"done {i}",
                completed_at=now + i,
                outcome="ok",
            )
        )
    completed = memory.list_completed(limit=3)
    assert len(completed) == 3
    # Most recent first
    assert completed[0].id == "g4"


def test_list_completed_excludes_active(memory):
    memory.save_goal(_make_goal("active", "still working"))
    memory.save_goal(
        _make_goal("done", "finished", completed_at=time.time(), outcome="ok")
    )
    completed = memory.list_completed()
    assert len(completed) == 1
    assert completed[0].id == "done"


# -- update_status --------------------------------------------------------

def test_update_status(memory):
    memory.save_goal(_make_goal("g1"))
    memory.update_status("g1", "failed")
    loaded = memory.get_goal("g1")
    # The raw status column is updated; on re-read the status is from the row
    # We check via list queries
    assert memory.list_active() == []  # no longer active


# -- delete ---------------------------------------------------------------

def test_delete_goal(memory):
    memory.save_goal(_make_goal("g1"))
    memory.delete_goal("g1")
    assert memory.get_goal("g1") is None


# -- edge cases -----------------------------------------------------------

def test_goal_without_chain_or_outcome(memory):
    goal = _make_goal("bare", "bare goal")
    memory.save_goal(goal)
    loaded = memory.get_goal("bare")
    assert loaded.thought_chain is None
    assert loaded.outcome is None
    assert loaded.lessons_learned == []
