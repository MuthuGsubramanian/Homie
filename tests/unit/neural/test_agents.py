# tests/unit/neural/test_agents.py
"""Tests for the four specialized agents."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from homie_core.neural.communication.agent_bus import AgentBus, AgentMessage
from homie_core.neural.agents.reasoning_agent import ReasoningAgent, ThoughtStep
from homie_core.neural.agents.research_agent import ResearchAgent
from homie_core.neural.agents.action_agent import ActionAgent
from homie_core.neural.agents.validation_agent import ValidationAgent


@pytest.fixture
def bus():
    b = AgentBus()
    yield b
    b.shutdown()


def _make_inference(response):
    """Return a MagicMock inference_fn that returns *response*."""
    return MagicMock(return_value=response)


# ── ReasoningAgent ──────────────────────────────────────────────────


class TestReasoningAgent:
    def test_analyze_valid_json(self, bus):
        llm_out = json.dumps({
            "analysis": "The query is about X",
            "confidence": 0.9,
            "key_findings": ["finding1"],
        })
        agent = ReasoningAgent(bus, _make_inference(llm_out))
        result = agent.analyze("What is X?", {"domain": "test"})
        assert result["confidence"] == 0.9
        assert "finding1" in result["key_findings"]

    def test_analyze_non_json_fallback(self, bus):
        agent = ReasoningAgent(bus, _make_inference("plain text answer"))
        result = agent.analyze("query")
        assert result["analysis"] == "plain text answer"
        assert result["confidence"] == 0.5

    def test_chain_of_thought_valid_json(self, bus):
        steps = [
            {"reasoning": "First gather data", "action": "research",
             "expected_outcome": "data collected", "agent": "research", "dependencies": []},
            {"reasoning": "Then analyze", "action": "analyze",
             "expected_outcome": "insights", "agent": "reasoning", "dependencies": ["0"]},
        ]
        agent = ReasoningAgent(bus, _make_inference(json.dumps(steps)))
        result = agent.chain_of_thought("Build a report")
        assert len(result) == 2
        assert isinstance(result[0], ThoughtStep)
        assert result[0].agent == "research"

    def test_chain_of_thought_fallback(self, bus):
        agent = ReasoningAgent(bus, _make_inference("not json"))
        result = agent.chain_of_thought("do something")
        assert len(result) == 1
        assert result[0].action == "do something"


# ── ResearchAgent ───────────────────────────────────────────────────


class TestResearchAgent:
    def test_search_knowledge(self, bus):
        data = [{"source": "kg", "content": "Python is a language", "relevance": 0.95}]
        agent = ResearchAgent(bus, _make_inference(json.dumps(data)))
        result = agent.search_knowledge("Python")
        assert len(result) == 1
        assert result[0]["source"] == "kg"

    def test_search_codebase(self, bus):
        data = [{"file_path": "main.py", "snippet": "def main():", "relevance": 0.8}]
        agent = ResearchAgent(bus, _make_inference(json.dumps(data)))
        result = agent.search_codebase("entry point")
        assert result[0]["file_path"] == "main.py"

    def test_gather_context(self, bus):
        ctx = {
            "summary": "overview",
            "knowledge_items": ["item1"],
            "relevant_files": [],
            "recommendations": ["do X"],
        }
        agent = ResearchAgent(bus, _make_inference(json.dumps(ctx)))
        result = agent.gather_context("build feature")
        assert result["summary"] == "overview"

    def test_gather_context_fallback(self, bus):
        agent = ResearchAgent(bus, _make_inference("raw text"))
        result = agent.gather_context("goal")
        assert result["summary"] == "raw text"
        assert result["knowledge_items"] == []


# ── ActionAgent ─────────────────────────────────────────────────────


class TestActionAgent:
    def test_execute_valid_json(self, bus):
        out = json.dumps({"status": "success", "output": "file created", "error": None})
        agent = ActionAgent(bus, _make_inference(out))
        result = agent.execute({"type": "file", "description": "create test.txt"})
        assert result["status"] == "success"
        assert result["output"] == "file created"

    def test_execute_non_json_fallback(self, bus):
        agent = ActionAgent(bus, _make_inference("done"))
        result = agent.execute({"type": "tool", "description": "run tool"})
        assert result["status"] == "success"
        assert result["output"] == "done"


# ── ValidationAgent ─────────────────────────────────────────────────


class TestValidationAgent:
    def test_validate_pass(self, bus):
        out = json.dumps({
            "valid": True,
            "score": 0.95,
            "issues": [],
            "suggestions": ["could add tests"],
        })
        agent = ValidationAgent(bus, _make_inference(out))
        result = agent.validate("build feature", {"output": "done"})
        assert result["valid"] is True
        assert result["score"] == 0.95
        assert result["issues"] == []

    def test_validate_fail(self, bus):
        out = json.dumps({
            "valid": False,
            "score": 0.2,
            "issues": ["missing error handling"],
            "suggestions": ["add try/except"],
        })
        agent = ValidationAgent(bus, _make_inference(out))
        result = agent.validate("handle errors", {"code": "pass"})
        assert result["valid"] is False
        assert len(result["issues"]) == 1

    def test_validate_non_json_fallback(self, bus):
        agent = ValidationAgent(bus, _make_inference("cannot parse"))
        result = agent.validate("goal", {})
        assert result["valid"] is False
        assert result["score"] == 0.0
        assert len(result["issues"]) == 1


# ── process() integration (async) ──────────────────────────────────


class TestAgentProcess:
    def test_reasoning_process(self, bus):
        out = json.dumps({"analysis": "deep", "confidence": 0.8, "key_findings": []})
        agent = ReasoningAgent(bus, _make_inference(out))
        msg = AgentMessage(
            from_agent="meta", to_agent="reasoning",
            message_type="goal", content={"action": "analyze", "query": "test"},
        )
        resp = asyncio.run(agent.process(msg))
        assert resp.to_agent == "meta"
        assert resp.message_type == "result"

    def test_validation_process(self, bus):
        out = json.dumps({"valid": True, "score": 1.0, "issues": [], "suggestions": []})
        agent = ValidationAgent(bus, _make_inference(out))
        msg = AgentMessage(
            from_agent="meta", to_agent="validation",
            message_type="goal", content={"goal": "test", "result": {"ok": True}},
        )
        resp = asyncio.run(agent.process(msg))
        assert resp.content["valid"] is True
