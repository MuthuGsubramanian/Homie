"""ResearchAgent — knowledge gathering from graphs, codebase, and documents."""

from __future__ import annotations

import json
import logging
from typing import Callable

from ..communication.agent_bus import AgentBus, AgentMessage
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """Gathers information from knowledge graph, codebase, and other sources."""

    def __init__(self, agent_bus: AgentBus, inference_fn: Callable) -> None:
        super().__init__(name="research", agent_bus=agent_bus, inference_fn=inference_fn)

    async def process(self, message: AgentMessage) -> AgentMessage:
        action = message.content.get("action", "gather_context")
        if action == "search_knowledge":
            result = self.search_knowledge(message.content.get("query", ""))
        elif action == "search_codebase":
            result = self.search_codebase(message.content.get("query", ""))
        elif action == "gather_context":
            result = self.gather_context(message.content.get("goal", ""))
        else:
            result = {"error": f"unknown action: {action}"}

        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            message_type="result",
            content=result if isinstance(result, dict) else {"results": result},
            parent_goal_id=message.parent_goal_id,
        )

    def search_knowledge(self, query: str) -> list[dict]:
        """Search the knowledge graph for relevant information."""
        prompt = (
            f"Search the knowledge graph for: {query}\n"
            f"Return a JSON array of relevant knowledge items with keys: "
            f"source, content, relevance."
        )
        raw = self.inference_fn(prompt)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [{"source": "llm", "content": raw, "relevance": 0.5}]

    def search_codebase(self, query: str) -> list[dict]:
        """Search the codebase for relevant files and code."""
        prompt = (
            f"Search the codebase for: {query}\n"
            f"Return a JSON array of relevant files with keys: "
            f"file_path, snippet, relevance."
        )
        raw = self.inference_fn(prompt)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [{"file_path": "unknown", "snippet": raw, "relevance": 0.5}]

    def gather_context(self, goal: str) -> dict:
        """Comprehensive context gathering for a goal — combines all sources."""
        prompt = (
            f"Gather comprehensive context for this goal: {goal}\n"
            f"Return a JSON object with keys: summary, knowledge_items, "
            f"relevant_files, recommendations."
        )
        raw = self.inference_fn(prompt)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {
                "summary": raw,
                "knowledge_items": [],
                "relevant_files": [],
                "recommendations": [],
            }
