"""Tests for web tool registration."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from homie_core.brain.tool_registry import ToolRegistry
from homie_core.web.models import WebPageAnalysis
from homie_core.web.tools import register_web_tools


class TestWebToolRegistration:
    def test_registers_web_analyze_tool(self) -> None:
        registry = ToolRegistry()
        register_web_tools(registry, MagicMock())
        tool = registry.get("web_analyze")
        assert tool is not None
        assert tool.name == "web_analyze"
        assert tool.category == "web"


class TestWebAnalyzeTool:
    def test_returns_json(self) -> None:
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_url.return_value = WebPageAnalysis(
            url="https://example.com",
            title="Test",
            page_type="webpage",
            description="desc",
            main_content="content",
            headings=["h1"],
            links_count=1,
            images_count=0,
            og_data={},
            analyzed_at=1000.0,
        )
        registry = ToolRegistry()
        register_web_tools(registry, mock_analyzer)
        tool = registry.get("web_analyze")
        output = tool.execute(url="https://example.com")
        data = json.loads(output)
        assert data["url"] == "https://example.com"
        assert data["title"] == "Test"
        assert data["page_type"] == "webpage"
