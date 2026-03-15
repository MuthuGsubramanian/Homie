from __future__ import annotations

import re

from homie_core.middleware.base import HomieMiddleware
from homie_core.backend.protocol import BackendProtocol
from homie_core.config import HomieConfig


class LargeResultEvictionMiddleware(HomieMiddleware):
    name = "large_result_eviction"
    order = 90
    EXCLUDED_TOOLS = {"ls", "glob", "grep", "read_file", "edit_file", "write_file", "search_files"}

    def __init__(self, config: HomieConfig, backend: BackendProtocol):
        self._threshold = config.context.large_result_threshold
        self._backend = backend

    def wrap_tool_result(self, name: str, result: str) -> str:
        if name in self.EXCLUDED_TOOLS:
            return result
        if len(result) <= self._threshold:
            return result
        safe_name = re.sub(r'[^\w\-.]', '_', name)
        path = f"/large_tool_results/{safe_name}_{id(result)}.md"
        self._backend.write(path, result)
        lines = result.splitlines()
        head = "\n".join(lines[:5])
        tail = "\n".join(lines[-5:])
        return (
            f"{head}\n\n"
            f"... ({len(lines)} total lines, {len(result)} chars) ...\n"
            f"Full output saved to: {path}\n\n"
            f"{tail}"
        )
