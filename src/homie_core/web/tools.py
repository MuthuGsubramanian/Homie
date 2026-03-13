from __future__ import annotations

import json

from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry

_MAX_OUTPUT = 2000


def _truncate(text: str) -> str:
    return text[:_MAX_OUTPUT] + "..." if len(text) > _MAX_OUTPUT else text


def register_web_tools(registry: ToolRegistry, web_analyzer) -> None:
    def tool_web_analyze(url: str) -> str:
        result = web_analyzer.analyze_url(url)
        return _truncate(json.dumps(result.to_dict()))

    registry.register(Tool(
        name="web_analyze",
        description="Fetch and analyze a webpage — extracts title, content, headings, links, images, and metadata.",
        params=[ToolParam(name="url", description="URL to analyze", type="string")],
        execute=tool_web_analyze,
        category="web",
    ))
