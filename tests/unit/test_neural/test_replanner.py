"""Tests for the Replanner — re-planning on step failure with mocked LLM."""

import json
import pytest

from homie_core.neural.planning.goal import ThoughtChain, ThoughtStep
from homie_core.neural.planning.replanner import Replanner


# -- helpers --------------------------------------------------------------

def _make_chain() -> ThoughtChain:
    return ThoughtChain(
        goal="deploy feature",
        steps=[
            ThoughtStep("s1", "build", "compile code", "binary", "action", []),
            ThoughtStep("s2", "test", "run tests", "pass", "validation", ["s1"]),
            ThoughtStep("s3", "deploy", "push to prod", "live", "action", ["s2"]),
        ],
    )


ALTERNATIVE_STEPS = [
    {
        "id": "s2-alt",
        "reasoning": "Try a different test approach",
        "action": "run smoke tests only",
        "expected_outcome": "smoke pass",
        "agent": "validation",
        "dependencies": ["s1"],
    },
    {
        "id": "s3-alt",
        "reasoning": "Deploy after smoke",
        "action": "push to staging first",
        "expected_outcome": "staging ok",
        "agent": "action",
        "dependencies": ["s2-alt"],
    },
]


def _mock_replan_fn(new_steps: list[dict]):
    def infer(prompt: str) -> str:
        return json.dumps({"steps": new_steps})
    return infer


# -- tests ----------------------------------------------------------------

def test_replan_basic():
    chain = _make_chain()
    chain.steps[0].status = "complete"
    chain.steps[1].status = "failed"

    replanner = Replanner(inference_fn=_mock_replan_fn(ALTERNATIVE_STEPS))
    new_chain = replanner.replan(chain, chain.steps[1], "tests timed out")

    assert new_chain.goal == chain.goal
    assert new_chain.status == "replanning"
    # Should include the completed step s1 + 2 new steps
    assert len(new_chain.steps) == 3
    assert new_chain.steps[0].id == "s1"
    assert new_chain.steps[0].status == "complete"
    assert new_chain.steps[1].id == "s2-alt"
    assert new_chain.steps[1].status == "pending"


def test_replan_preserves_completed_steps():
    chain = _make_chain()
    chain.steps[0].status = "complete"
    chain.steps[0].result = {"binary": "/tmp/app"}
    chain.steps[1].status = "failed"

    replanner = Replanner(inference_fn=_mock_replan_fn(ALTERNATIVE_STEPS))
    new_chain = replanner.replan(chain, chain.steps[1], "error")

    assert new_chain.steps[0].result == {"binary": "/tmp/app"}


def test_replan_budget_enforced():
    chain = _make_chain()
    chain.steps[0].status = "complete"
    chain.steps[1].status = "failed"

    replanner = Replanner(
        inference_fn=_mock_replan_fn(ALTERNATIVE_STEPS), max_replans=2
    )

    # Two replans should succeed
    replanner.replan(chain, chain.steps[1], "err1")
    replanner.replan(chain, chain.steps[1], "err2")

    # Third should raise
    with pytest.raises(RuntimeError, match="budget exhausted"):
        replanner.replan(chain, chain.steps[1], "err3")


def test_can_replan():
    chain = _make_chain()
    chain.steps[0].status = "complete"
    chain.steps[1].status = "failed"

    replanner = Replanner(
        inference_fn=_mock_replan_fn(ALTERNATIVE_STEPS), max_replans=1
    )
    assert replanner.can_replan(chain)
    replanner.replan(chain, chain.steps[1], "err")
    assert not replanner.can_replan(chain)


def test_reset_replan_counter():
    chain = _make_chain()
    chain.steps[0].status = "complete"
    chain.steps[1].status = "failed"

    replanner = Replanner(
        inference_fn=_mock_replan_fn(ALTERNATIVE_STEPS), max_replans=1
    )
    replanner.replan(chain, chain.steps[1], "err")
    assert not replanner.can_replan(chain)

    replanner.reset(chain.goal)
    assert replanner.can_replan(chain)


def test_replan_with_markdown_fenced_json():
    chain = _make_chain()
    chain.steps[0].status = "complete"
    chain.steps[1].status = "failed"

    fenced = "```json\n" + json.dumps({"steps": ALTERNATIVE_STEPS}) + "\n```"

    def infer(prompt: str) -> str:
        return fenced

    replanner = Replanner(inference_fn=infer)
    new_chain = replanner.replan(chain, chain.steps[1], "error")
    assert len(new_chain.steps) == 3


def test_max_replans_property():
    replanner = Replanner(inference_fn=lambda p: "", max_replans=7)
    assert replanner.max_replans == 7
