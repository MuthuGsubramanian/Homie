"""ValidationAgent — verifies results against goals and scores quality."""

from __future__ import annotations

import json
import logging
from typing import Callable

from ..communication.agent_bus import AgentBus, AgentMessage
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """Read-only agent that checks whether a result meets the stated goal."""

    def __init__(self, agent_bus: AgentBus, inference_fn: Callable) -> None:
        super().__init__(name="validation", agent_bus=agent_bus, inference_fn=inference_fn)

    async def process(self, message: AgentMessage) -> AgentMessage:
        result = self.validate(
            goal=message.content.get("goal", ""),
            result=message.content.get("result", {}),
        )
        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            message_type="result",
            content=result,
            parent_goal_id=message.parent_goal_id,
        )

    def validate(self, goal: str, result: dict) -> dict:
        """Validate whether *result* satisfies *goal*.

        Returns a dict with:
        - ``valid`` (bool)
        - ``score`` (float 0-1)
        - ``issues`` (list[str])
        - ``suggestions`` (list[str])
        """
        prompt = (
            f"Validate whether this result meets the goal.\n"
            f"Goal: {goal}\n"
            f"Result: {json.dumps(result)}\n"
            f"Return JSON with keys: valid (bool), score (0-1), "
            f"issues (list of strings), suggestions (list of strings)."
        )
        raw = self.inference_fn(prompt)
        try:
            parsed = json.loads(raw)
            return {
                "valid": parsed.get("valid", False),
                "score": float(parsed.get("score", 0.0)),
                "issues": parsed.get("issues", []),
                "suggestions": parsed.get("suggestions", []),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            return {
                "valid": False,
                "score": 0.0,
                "issues": ["Unable to parse validation output from LLM."],
                "suggestions": [],
            }
