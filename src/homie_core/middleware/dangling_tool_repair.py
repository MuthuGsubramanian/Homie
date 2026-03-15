from __future__ import annotations
import re
from homie_core.middleware.base import HomieMiddleware
from homie_core.memory.working import WorkingMemory

# Matches Homie's tool call format: <tool>name(...)</tool>
_TOOL_CALL_RE = re.compile(r'<tool>\s*(\w+)\s*\([^)]*\)\s*</tool>', re.DOTALL)
# Matches tool result format: [Tool: name] Result: ...  OR  [Tool: name] Error: ...
_TOOL_RESULT_RE = re.compile(r'\[Tool:\s*(\w+)\]\s*(?:Result|Error):', re.DOTALL)


class DanglingToolCallMiddleware(HomieMiddleware):
    name = "dangling_tool_repair"
    order = 2

    def __init__(self, working_memory: WorkingMemory):
        self._wm = working_memory

    def before_turn(self, message: str, state: dict) -> str:
        conversation = self._wm.get_conversation()
        if len(conversation) < 2:
            return message

        repaired = list(conversation)
        insertions = []

        for i, msg in enumerate(repaired):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            tool_calls = _TOOL_CALL_RE.findall(content)
            if not tool_calls:
                continue

            # Check if the next message contains a tool result for each call
            next_msg = repaired[i + 1] if i + 1 < len(repaired) else None
            next_content = next_msg.get("content", "") if next_msg else ""
            result_tools = set(_TOOL_RESULT_RE.findall(next_content)) if next_msg else set()

            for tool_name in tool_calls:
                if tool_name not in result_tools:
                    # This tool call has no result — inject synthetic one
                    insertions.append((i + 1, {
                        "role": "system",
                        "content": (
                            f"[Tool: {tool_name}] Result: Tool call was cancelled"
                            " \u2014 another message arrived before it could complete."
                        ),
                    }))

        # Apply insertions in reverse order to maintain indices
        for idx, msg in sorted(insertions, reverse=True):
            repaired.insert(idx, msg)

        if insertions:
            self._wm._conversation.clear()
            self._wm._conversation.extend(repaired)

        return message
