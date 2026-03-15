from __future__ import annotations

from homie_core.middleware.base import HomieMiddleware
from homie_core.memory.working import WorkingMemory
from homie_core.config import HomieConfig


class ArgTruncationMiddleware(HomieMiddleware):
    name = "arg_truncation"
    order = 8
    TRUNCATABLE_TOOLS = {"write_file", "edit_file"}

    def __init__(self, config: HomieConfig, working_memory: WorkingMemory):
        self._threshold = config.context.arg_truncation_threshold
        self._wm = working_memory

    def before_turn(self, message: str, state: dict) -> str:
        conversation = self._wm.get_conversation()
        if len(conversation) < 6:
            return message
        for msg in conversation[:-3]:
            content = msg.get("content", "")
            if not content or msg.get("role") != "assistant":
                continue
            if len(content) <= self._threshold:
                continue
            if self._has_truncatable_tool(content):
                msg["content"] = content[:20] + "...(argument truncated)"
        return message

    def _has_truncatable_tool(self, content: str) -> bool:
        lower = content.lower()
        return any(tool in lower for tool in self.TRUNCATABLE_TOOLS)
