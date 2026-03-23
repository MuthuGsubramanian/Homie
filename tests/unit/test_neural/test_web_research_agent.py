"""Unit tests for WebResearchAgent."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from homie_core.neural.agents.web_research_agent import WebResearchAgent
from homie_core.neural.communication.agent_bus import AgentBus, AgentMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def bus():
    b = AgentBus()
    yield b
    b.shutdown()


def _inference_fn(response):
    def fn(prompt: str, **kw) -> str:
        return response
    return fn


def _json_inference(obj):
    return _inference_fn(json.dumps(obj))


def _make_knowledge_store(results=None):
    store = MagicMock()
    store.query.return_value = results or []
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResearch:
    def test_research_returns_structured_result(self, bus):
        resp = {
            "summary": "Python is a programming language.",
            "key_findings": ["Created by Guido van Rossum"],
            "sources_used": ["general knowledge"],
            "confidence": 0.85,
            "gaps": [],
            "related_topics": ["CPython", "PyPy"],
        }
        agent = WebResearchAgent(bus, _json_inference(resp))
        result = agent.research("Python programming language")
        assert result["summary"] == "Python is a programming language."
        assert result["confidence"] == 0.85
        assert "CPython" in result["related_topics"]

    def test_research_empty_topic(self, bus):
        agent = WebResearchAgent(bus, _json_inference({}))
        result = agent.research("")
        assert result["confidence"] == 0.0
        assert "No topic provided" in result["gaps"]

    def test_research_uses_knowledge_store(self, bus):
        store = _make_knowledge_store(["Python was created in 1991"])
        resp = {
            "summary": "Python was created in 1991.",
            "key_findings": [],
            "sources_used": ["knowledge graph"],
            "confidence": 0.9,
            "gaps": [],
            "related_topics": [],
        }
        agent = WebResearchAgent(bus, _json_inference(resp), knowledge_store=store)
        result = agent.research("Python history")
        store.query.assert_called_once_with("Python history")
        assert result["confidence"] == 0.9

    def test_research_handles_llm_failure(self, bus):
        def failing(prompt, **kw):
            raise RuntimeError("LLM down")
        agent = WebResearchAgent(bus, failing)
        result = agent.research("anything")
        assert result["confidence"] == 0.0

    def test_research_clamps_confidence(self, bus):
        resp = {"summary": "test", "confidence": 5.0}
        agent = WebResearchAgent(bus, _json_inference(resp))
        result = agent.research("test topic")
        assert result["confidence"] == 1.0

    def test_research_with_depth(self, bus):
        resp = {"summary": "deep analysis", "confidence": 0.7}
        agent = WebResearchAgent(bus, _json_inference(resp))
        result = agent.research("quantum computing", depth="deep")
        assert result["summary"] == "deep analysis"


class TestFactCheck:
    def test_fact_check_true_claim(self, bus):
        resp = {
            "verdict": "true",
            "confidence": 0.95,
            "explanation": "This is correct.",
            "supporting_evidence": ["Source A"],
            "contradicting_evidence": [],
            "caveats": [],
        }
        agent = WebResearchAgent(bus, _json_inference(resp))
        result = agent.fact_check("The Earth orbits the Sun")
        assert result["verdict"] == "true"
        assert result["confidence"] == 0.95

    def test_fact_check_empty_claim(self, bus):
        agent = WebResearchAgent(bus, _json_inference({}))
        result = agent.fact_check("")
        assert result["verdict"] == "unverifiable"
        assert result["confidence"] == 0.0

    def test_fact_check_invalid_verdict_normalized(self, bus):
        resp = {"verdict": "maybe", "confidence": 0.5, "explanation": "Unclear"}
        agent = WebResearchAgent(bus, _json_inference(resp))
        result = agent.fact_check("Some ambiguous claim")
        assert result["verdict"] == "unverifiable"

    def test_fact_check_with_knowledge_store(self, bus):
        store = _make_knowledge_store(["Water boils at 100C at sea level"])
        resp = {
            "verdict": "partially_true",
            "confidence": 0.8,
            "explanation": "Depends on altitude.",
        }
        agent = WebResearchAgent(bus, _json_inference(resp), knowledge_store=store)
        result = agent.fact_check("Water always boils at 100C")
        assert result["verdict"] == "partially_true"
        store.query.assert_called_once()


class TestSummarizeSources:
    def test_summarize_multiple_sources(self, bus):
        agent = WebResearchAgent(bus, _inference_fn("Combined summary of all sources."))
        result = agent.summarize_sources(["Source 1 text", "Source 2 text"])
        assert "summary" in result.lower()

    def test_summarize_empty_sources(self, bus):
        agent = WebResearchAgent(bus, _inference_fn("should not be called"))
        assert agent.summarize_sources([]) == "No sources provided."

    def test_summarize_handles_failure(self, bus):
        def failing(prompt, **kw):
            raise RuntimeError("LLM down")
        agent = WebResearchAgent(bus, failing)
        result = agent.summarize_sources(["some source"])
        assert "failed" in result.lower()


class TestProcessMessage:
    def test_process_routes_research(self, bus):
        resp = {"summary": "result", "confidence": 0.7}
        agent = WebResearchAgent(bus, _json_inference(resp))
        msg = AgentMessage(
            from_agent="meta", to_agent="web_research",
            message_type="goal",
            content={"action": "research", "topic": "AI safety"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent.process(msg))
        assert result.content["summary"] == "result"

    def test_process_routes_fact_check(self, bus):
        resp = {"verdict": "true", "confidence": 0.9, "explanation": "correct"}
        agent = WebResearchAgent(bus, _json_inference(resp))
        msg = AgentMessage(
            from_agent="meta", to_agent="web_research",
            message_type="goal",
            content={"action": "fact_check", "claim": "Earth is round"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent.process(msg))
        assert result.content["verdict"] == "true"

    def test_process_routes_unknown_action(self, bus):
        agent = WebResearchAgent(bus, _json_inference({}))
        msg = AgentMessage(
            from_agent="meta", to_agent="web_research",
            message_type="goal",
            content={"action": "dance"},
        )
        result = asyncio.get_event_loop().run_until_complete(agent.process(msg))
        assert "error" in result.content
