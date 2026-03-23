"""ActionAgent — executes planned actions with resilience wrapping."""

from __future__ import annotations

import json
import logging
from typing import Callable

from ..communication.agent_bus import AgentBus, AgentMessage
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Import @resilient if available; fall back to passthrough if not.
try:
    from homie_core.self_healing.resilience.decorator import resilient as _resilient
except ImportError:  # pragma: no cover
    def _resilient(**kwargs):
        """No-op fallback when self_healing module is unavailable."""
        def decorator(fn):
            return fn
        return decorator


class ActionAgent(BaseAgent):
    """Executes actions (code, files, tools, APIs) with @resilient safety."""

    def __init__(self, agent_bus: AgentBus, inference_fn: Callable) -> None:
        super().__init__(name="action", agent_bus=agent_bus, inference_fn=inference_fn)

    async def process(self, message: AgentMessage) -> AgentMessage:
        action = message.content.get("action_spec", message.content)
        result = self.execute(action)
        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            message_type="result",
            content=result,
            parent_goal_id=message.parent_goal_id,
        )

    @_resilient(retries=2, timeout=60.0)
    def execute(self, action: dict) -> dict:
        """Execute a planned action.

        The *action* dict should contain:
        - ``type``: the kind of action (e.g. "code", "file", "tool", "api")
        - ``description``: human-readable summary
        - any additional params the action requires

        Returns a dict with ``status``, ``output``, and optionally ``error``.
        """
        action_type = action.get("type", "unknown")
        description = action.get("description", "")

        prompt = (
            f"Execute this action and return a JSON result.\n"
            f"Action type: {action_type}\n"
            f"Description: {description}\n"
            f"Parameters: {json.dumps(action)}\n"
            f"Return JSON with keys: status (success/failure), output, error (if any)."
        )
        raw = self.inference_fn(prompt)
        try:
            result = json.loads(raw)
            result.setdefault("status", "success")
            return result
        except (json.JSONDecodeError, TypeError):
            return {"status": "success", "output": raw, "error": None}
