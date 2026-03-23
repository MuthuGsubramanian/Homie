"""ReasoningAgent — deep analysis and chain-of-thought reasoning."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

from ..communication.agent_bus import AgentBus, AgentMessage
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class ThoughtStep:
    """One step in a chain-of-thought plan."""

    reasoning: str
    action: str
    expected_outcome: str
    agent: str
    dependencies: list[str] = field(default_factory=list)
    result: dict | None = None
    status: str = "pending"  # pending, active, complete, failed


class ReasoningAgent(BaseAgent):
    """Uses LLM inference to perform deep analysis and structured reasoning."""

    def __init__(self, agent_bus: AgentBus, inference_fn: Callable) -> None:
        super().__init__(name="reasoning", agent_bus=agent_bus, inference_fn=inference_fn)

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Dispatch based on message content."""
        action = message.content.get("action", "analyze")
        if action == "analyze":
            result = self.analyze(
                message.content.get("query", ""),
                message.content.get("context", {}),
            )
        elif action == "chain_of_thought":
            steps = self.chain_of_thought(message.content.get("goal", ""))
            result = {"steps": [s.__dict__ for s in steps]}
        else:
            result = {"error": f"unknown action: {action}"}

        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            message_type="result",
            content=result,
            parent_goal_id=message.parent_goal_id,
        )

    def analyze(self, query: str, context: dict | None = None) -> dict:
        """Use the LLM for deep analysis of *query* with optional *context*."""
        prompt = (
            f"Analyze the following query deeply.\n"
            f"Query: {query}\n"
            f"Context: {json.dumps(context or {})}\n"
            f"Return a JSON object with keys: analysis, confidence, key_findings."
        )
        raw = self.inference_fn(prompt)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"analysis": raw, "confidence": 0.5, "key_findings": []}

    def chain_of_thought(self, goal: str) -> list[ThoughtStep]:
        """Generate a structured chain-of-thought plan for *goal*."""
        prompt = (
            f"Break down this goal into a chain-of-thought plan.\n"
            f"Goal: {goal}\n"
            f"Return a JSON array of objects with keys: reasoning, action, "
            f"expected_outcome, agent, dependencies."
        )
        raw = self.inference_fn(prompt)
        try:
            items = json.loads(raw)
            return [
                ThoughtStep(
                    reasoning=item.get("reasoning", ""),
                    action=item.get("action", ""),
                    expected_outcome=item.get("expected_outcome", ""),
                    agent=item.get("agent", "reasoning"),
                    dependencies=item.get("dependencies", []),
                )
                for item in items
            ]
        except (json.JSONDecodeError, TypeError):
            return [
                ThoughtStep(
                    reasoning="Unable to parse LLM output — treating goal as single step.",
                    action=goal,
                    expected_outcome="Goal completed",
                    agent="reasoning",
                )
            ]
