from __future__ import annotations

from typing import Any, Optional


class HomieMiddleware:
    name: str = "unnamed"
    order: int = 100

    def modify_tools(self, tools: list) -> list:
        return tools

    def modify_prompt(self, prompt: str) -> str:
        return prompt

    def before_turn(self, message: str, state: dict) -> str | None:
        return message

    def after_turn(self, response: str, state: dict) -> str:
        return response

    def wrap_tool_call(self, name: str, args: dict) -> dict | None:
        return args

    def wrap_tool_result(self, name: str, result: Any) -> Any:
        return result
