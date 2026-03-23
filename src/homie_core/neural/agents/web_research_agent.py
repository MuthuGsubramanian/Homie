"""WebResearchAgent — research topics, fact-check claims, synthesize sources."""

from __future__ import annotations

import json
import logging
from typing import Callable

from ..communication.agent_bus import AgentBus, AgentMessage
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ── LLM prompts ──────────────────────────────────────────────────────────────

_RESEARCH_PROMPT = """\
Research the following topic thoroughly. Return ONLY a JSON object.

Topic: {topic}
Depth: {depth}

{context_block}

Return: {{"summary": "<comprehensive summary>", "key_findings": [<list of important findings>], "sources_used": [<list of source descriptions>], "confidence": <float 0.0-1.0>, "gaps": [<list of knowledge gaps or uncertainties>], "related_topics": [<list of related topics worth exploring>]}}"""

_FACT_CHECK_PROMPT = """\
Fact-check the following claim. Return ONLY a JSON object.

Claim: {claim}

{context_block}

Return: {{"verdict": "<true|false|partially_true|unverifiable>", "confidence": <float 0.0-1.0>, "explanation": "<why this verdict>", "supporting_evidence": [<evidence supporting the claim>], "contradicting_evidence": [<evidence against the claim>], "caveats": [<important nuances or conditions>]}}"""

_SUMMARIZE_PROMPT = """\
Summarize the following {count} sources into a coherent overview. Be concise and factual.

{sources_block}

Return a clear, well-structured summary that synthesizes the key points across all sources."""


def _parse_json(raw: str) -> dict | list | None:
    """Extract JSON from LLM output, handling markdown fences."""
    import re
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    for i, c in enumerate(cleaned):
        if c in "{[":
            cleaned = cleaned[i:]
            break
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


class WebResearchAgent(BaseAgent):
    """Research agent that synthesizes knowledge from available sources.

    Uses inference_fn for all LLM calls. Optionally queries a knowledge
    graph if one is provided.
    """

    def __init__(
        self,
        agent_bus: AgentBus,
        inference_fn: Callable,
        knowledge_store=None,
    ) -> None:
        super().__init__(name="web_research", agent_bus=agent_bus, inference_fn=inference_fn)
        self._knowledge_store = knowledge_store

    # ── BaseAgent interface ──────────────────────────────────────────

    async def process(self, message: AgentMessage) -> AgentMessage:
        action = message.content.get("action", "research")

        if action == "research":
            result = self.research(
                message.content.get("topic", ""),
                message.content.get("depth", "moderate"),
            )
        elif action == "fact_check":
            result = self.fact_check(message.content.get("claim", ""))
        elif action == "summarize_sources":
            result = {"summary": self.summarize_sources(message.content.get("sources", []))}
        else:
            result = {"error": f"unknown action: {action}"}

        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            message_type="result",
            content=result if isinstance(result, dict) else {"result": result},
            parent_goal_id=message.parent_goal_id,
        )

    # ── Public API ───────────────────────────────────────────────────

    def research(self, topic: str, depth: str = "moderate") -> dict:
        """Research a topic using available knowledge + LLM inference.

        Args:
            topic: The topic to research.
            depth: One of "shallow", "moderate", "deep".

        Returns:
            Dict with summary, key_findings, confidence, gaps, etc.
        """
        if not topic:
            return {
                "summary": "",
                "key_findings": [],
                "sources_used": [],
                "confidence": 0.0,
                "gaps": ["No topic provided"],
                "related_topics": [],
            }

        # Gather context from knowledge graph if available
        context_items = self._query_knowledge(topic)
        context_block = ""
        if context_items:
            context_block = "Known context:\n" + "\n".join(
                f"- {item}" for item in context_items
            )

        prompt = _RESEARCH_PROMPT.format(
            topic=topic,
            depth=depth,
            context_block=context_block,
        )

        try:
            raw = self.inference_fn(prompt)
            parsed = _parse_json(raw)
            if isinstance(parsed, dict) and "summary" in parsed:
                parsed.setdefault("key_findings", [])
                parsed.setdefault("sources_used", [])
                parsed.setdefault("confidence", 0.5)
                parsed.setdefault("gaps", [])
                parsed.setdefault("related_topics", [])
                parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
                return parsed
        except Exception as exc:
            logger.debug("Research failed: %s", exc)

        return {
            "summary": raw if "raw" in dir() else f"Research on '{topic}' failed.",
            "key_findings": [],
            "sources_used": [],
            "confidence": 0.0,
            "gaps": ["LLM parsing failed"],
            "related_topics": [],
        }

    def fact_check(self, claim: str) -> dict:
        """Verify a claim against known knowledge and LLM reasoning.

        Returns dict with verdict, confidence, explanation, evidence.
        """
        if not claim:
            return {
                "verdict": "unverifiable",
                "confidence": 0.0,
                "explanation": "No claim provided.",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "caveats": [],
            }

        context_items = self._query_knowledge(claim)
        context_block = ""
        if context_items:
            context_block = "Known context:\n" + "\n".join(
                f"- {item}" for item in context_items
            )

        prompt = _FACT_CHECK_PROMPT.format(claim=claim, context_block=context_block)

        try:
            raw = self.inference_fn(prompt)
            parsed = _parse_json(raw)
            if isinstance(parsed, dict) and "verdict" in parsed:
                valid_verdicts = {"true", "false", "partially_true", "unverifiable"}
                if parsed.get("verdict") not in valid_verdicts:
                    parsed["verdict"] = "unverifiable"
                parsed.setdefault("confidence", 0.5)
                parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
                parsed.setdefault("explanation", "")
                parsed.setdefault("supporting_evidence", [])
                parsed.setdefault("contradicting_evidence", [])
                parsed.setdefault("caveats", [])
                return parsed
        except Exception as exc:
            logger.debug("Fact check failed: %s", exc)

        return {
            "verdict": "unverifiable",
            "confidence": 0.0,
            "explanation": "Fact-checking failed.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "caveats": [],
        }

    def summarize_sources(self, sources: list[str]) -> str:
        """Summarize multiple information sources into a coherent overview."""
        if not sources:
            return "No sources provided."

        sources_block = "\n\n".join(
            f"--- Source {i + 1} ---\n{src[:2000]}"
            for i, src in enumerate(sources[:20])
        )

        prompt = _SUMMARIZE_PROMPT.format(
            count=len(sources),
            sources_block=sources_block,
        )

        try:
            return self.inference_fn(prompt)
        except Exception as exc:
            logger.debug("Source summarization failed: %s", exc)
            return f"Summarization failed: {exc}"

    # ── Private helpers ──────────────────────────────────────────────

    def _query_knowledge(self, query: str) -> list[str]:
        """Query knowledge store if available. Returns list of context strings."""
        if self._knowledge_store is None:
            return []
        try:
            results = self._knowledge_store.query(query)
            if isinstance(results, list):
                return [str(r) for r in results[:10]]
        except Exception as exc:
            logger.debug("Knowledge store query failed: %s", exc)
        return []
