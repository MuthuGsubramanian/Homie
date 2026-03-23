"""Tests for the Planner — goal decomposition with mocked LLM."""

import json

from homie_core.neural.planning.planner import Planner, COMPLEXITY_TO_STRATEGY


# -- helpers --------------------------------------------------------------

def _mock_classify(response: str):
    """Return an inference_fn that always returns *response*."""
    def infer(prompt: str) -> str:
        return response
    return infer


def _mock_plan_fn(steps_data: list[dict]):
    """Return an inference_fn that classifies as 'simple' on classify prompts
    and returns a plan JSON on plan prompts."""
    call_count = {"n": 0}

    def infer(prompt: str) -> str:
        call_count["n"] += 1
        if "Classify the complexity" in prompt:
            return "simple"
        return json.dumps({"steps": steps_data})

    return infer


SINGLE_STEP = [
    {
        "id": "step-1",
        "reasoning": "Just do it",
        "action": "say hello",
        "expected_outcome": "greeting delivered",
        "agent": "action",
        "dependencies": [],
    }
]

LINEAR_STEPS = [
    {
        "id": "step-1",
        "reasoning": "First gather info",
        "action": "search knowledge",
        "expected_outcome": "context loaded",
        "agent": "research",
        "dependencies": [],
    },
    {
        "id": "step-2",
        "reasoning": "Then analyze",
        "action": "reason about data",
        "expected_outcome": "insights found",
        "agent": "reasoning",
        "dependencies": ["step-1"],
    },
    {
        "id": "step-3",
        "reasoning": "Finally act",
        "action": "write report",
        "expected_outcome": "report generated",
        "agent": "action",
        "dependencies": ["step-2"],
    },
]

PARALLEL_STEPS = [
    {
        "id": "step-1",
        "reasoning": "Fetch emails",
        "action": "read inbox",
        "expected_outcome": "emails loaded",
        "agent": "research",
        "dependencies": [],
    },
    {
        "id": "step-2",
        "reasoning": "Fetch calendar",
        "action": "read calendar",
        "expected_outcome": "events loaded",
        "agent": "research",
        "dependencies": [],
    },
    {
        "id": "step-3",
        "reasoning": "Combine",
        "action": "generate briefing",
        "expected_outcome": "briefing ready",
        "agent": "action",
        "dependencies": ["step-1", "step-2"],
    },
]


# -- classify_goal_complexity tests ---------------------------------------

def test_classify_trivial():
    planner = Planner(inference_fn=_mock_classify("trivial"))
    assert planner.classify_goal_complexity("hello") == "trivial"


def test_classify_simple():
    planner = Planner(inference_fn=_mock_classify("simple"))
    assert planner.classify_goal_complexity("set a timer") == "simple"


def test_classify_moderate():
    planner = Planner(inference_fn=_mock_classify("moderate"))
    assert planner.classify_goal_complexity("analyze expenses") == "moderate"


def test_classify_complex():
    planner = Planner(inference_fn=_mock_classify("The goal is complex."))
    assert planner.classify_goal_complexity("prepare annual tax filing") == "complex"


def test_classify_fallback():
    planner = Planner(inference_fn=_mock_classify("I'm not sure"))
    assert planner.classify_goal_complexity("vague thing") == "moderate"


# -- plan tests -----------------------------------------------------------

def test_plan_single_step():
    planner = Planner(inference_fn=_mock_plan_fn(SINGLE_STEP))
    chain = planner.plan("greet the user")
    assert chain.goal == "greet the user"
    assert len(chain.steps) == 1
    assert chain.steps[0].agent == "action"
    assert chain.status == "thinking"


def test_plan_linear():
    planner = Planner(inference_fn=_mock_plan_fn(LINEAR_STEPS))
    chain = planner.plan("write a report")
    assert len(chain.steps) == 3
    assert chain.steps[1].dependencies == ["step-1"]
    assert chain.steps[2].dependencies == ["step-2"]


def test_plan_parallel():
    planner = Planner(inference_fn=_mock_plan_fn(PARALLEL_STEPS))
    chain = planner.plan("morning briefing")
    assert len(chain.steps) == 3
    ready = chain.get_ready_steps()
    assert len(ready) == 2  # step-1 and step-2 have no deps


def test_plan_with_context():
    planner = Planner(inference_fn=_mock_plan_fn(SINGLE_STEP))
    chain = planner.plan("greet", context={"user": "Alice", "time": "morning"})
    assert chain.goal == "greet"
    assert len(chain.steps) == 1


def test_plan_parses_markdown_fenced_json():
    """LLM wraps JSON in markdown code fences."""
    fenced = '```json\n' + json.dumps({"steps": SINGLE_STEP}) + '\n```'

    def infer(prompt: str) -> str:
        if "Classify" in prompt:
            return "trivial"
        return fenced

    planner = Planner(inference_fn=infer)
    chain = planner.plan("hi")
    assert len(chain.steps) == 1


def test_plan_parses_json_with_preamble():
    """LLM includes text before the JSON."""
    raw = 'Here is the plan:\n' + json.dumps({"steps": LINEAR_STEPS})

    def infer(prompt: str) -> str:
        if "Classify" in prompt:
            return "simple"
        return raw

    planner = Planner(inference_fn=infer)
    chain = planner.plan("do something")
    assert len(chain.steps) == 3


def test_complexity_to_strategy_mapping():
    assert COMPLEXITY_TO_STRATEGY["trivial"] == "direct"
    assert COMPLEXITY_TO_STRATEGY["simple"] == "linear"
    assert COMPLEXITY_TO_STRATEGY["moderate"] == "parallel"
    assert COMPLEXITY_TO_STRATEGY["complex"] == "hierarchical"
