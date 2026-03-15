from __future__ import annotations
from typing import Callable
from homie_core.middleware.base import HomieMiddleware


class SubAgentMiddleware(HomieMiddleware):
    name = "subagent"
    order = 50

    def __init__(self, agent_factory: Callable[[str], str]):
        """agent_factory: callable that takes a task description and returns a response string.
        The caller is responsible for creating the isolated agent — this middleware just
        provides the tool and delegates execution to the factory."""
        self._agent_factory = agent_factory

    def modify_tools(self, tools: list[dict]) -> list[dict]:
        task_tool = {
            "name": "task",
            "description": (
                "Spawn an isolated sub-agent to handle a complex task. "
                "The sub-agent gets a fresh context window with only your task description. "
                "Returns the sub-agent's response."
            ),
        }
        return tools + [task_tool]

    def wrap_tool_call(self, name: str, args: dict) -> dict | None:
        if name != "task":
            return args
        description = args.get("description", "")
        if not description:
            return args
        try:
            result = self._agent_factory(description)
            args["_subagent_result"] = result
        except Exception as e:
            args["_subagent_result"] = f"Sub-agent error: {e}"
        return args
