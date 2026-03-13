"""Tests for browser AI tool registration and execution."""
import json
import pytest
from unittest.mock import MagicMock

from homie_core.brain.tool_registry import ToolRegistry
from homie_core.browser.tools import register_browser_tools


class TestBrowserToolRegistration:
    def _setup(self):
        registry = ToolRegistry()
        browser_service = MagicMock()
        register_browser_tools(registry, browser_service)
        return registry, browser_service

    def test_registers_four_tools(self):
        registry, _ = self._setup()
        tools = registry.list_tools()
        names = [t.name for t in tools]
        assert "browser_history" in names
        assert "browser_patterns" in names
        assert "browser_scan" in names
        assert "browser_config" in names
        assert len(names) == 4

    def test_all_tools_browser_category(self):
        registry, _ = self._setup()
        for tool in registry.list_tools():
            assert tool.category == "browser"


class TestBrowserHistoryTool:
    def test_returns_json(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_history.return_value = [
            {"url": "https://example.com", "title": "Ex", "visit_time": 1700000000.0}
        ]
        register_browser_tools(registry, service)

        tool = registry.get("browser_history")
        result = tool.execute()
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["url"] == "https://example.com"

    def test_passes_limit(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_history.return_value = []
        register_browser_tools(registry, service)

        tool = registry.get("browser_history")
        tool.execute(limit="10")
        service.get_history.assert_called_with(limit=10, domain=None, since=None)

    def test_passes_domain(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_history.return_value = []
        register_browser_tools(registry, service)

        tool = registry.get("browser_history")
        tool.execute(domain="github.com")
        service.get_history.assert_called_with(limit=50, domain="github.com", since=None)

    def test_invalid_limit_defaults(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_history.return_value = []
        register_browser_tools(registry, service)

        tool = registry.get("browser_history")
        tool.execute(limit="not_a_number")
        service.get_history.assert_called_with(limit=50, domain=None, since=None)


class TestBrowserPatternsTool:
    def test_returns_json(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_patterns.return_value = {
            "top_domains": [], "peak_hours": [], "daily_avg_pages": 0,
            "category_breakdown": {},
        }
        register_browser_tools(registry, service)

        tool = registry.get("browser_patterns")
        result = tool.execute()
        parsed = json.loads(result)
        assert "top_domains" in parsed


class TestBrowserScanTool:
    def test_returns_json(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.scan.return_value = {"entries_count": 5, "patterns": {}}
        register_browser_tools(registry, service)

        tool = registry.get("browser_scan")
        result = tool.execute()
        parsed = json.loads(result)
        assert parsed["entries_count"] == 5


class TestBrowserConfigTool:
    def test_get_config(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_config.return_value = {"enabled": False, "browsers": ["chrome"]}
        register_browser_tools(registry, service)

        tool = registry.get("browser_config")
        result = tool.execute()
        parsed = json.loads(result)
        assert parsed["enabled"] is False

    def test_set_config(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.configure.return_value = {"enabled": True, "browsers": ["chrome", "firefox"]}
        register_browser_tools(registry, service)

        tool = registry.get("browser_config")
        result = tool.execute(enabled="true", browsers="chrome,firefox")
        service.configure.assert_called_once_with(
            enabled=True, browsers=["chrome", "firefox"]
        )

    def test_set_retention_days(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.configure.return_value = {"retention_days": 7}
        register_browser_tools(registry, service)

        tool = registry.get("browser_config")
        tool.execute(retention_days="7")
        service.configure.assert_called_once_with(retention_days=7)

    def test_set_exclude_domains(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.configure.return_value = {}
        register_browser_tools(registry, service)

        tool = registry.get("browser_config")
        tool.execute(exclude_domains="facebook.com, twitter.com")
        service.configure.assert_called_once_with(
            exclude_domains=["facebook.com", "twitter.com"]
        )

    def test_invalid_retention_ignored(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_config.return_value = {"enabled": False}
        register_browser_tools(registry, service)

        tool = registry.get("browser_config")
        # Only retention_days passed but invalid - should still call get_config
        # because no valid kwargs end up being set... actually retention_days="abc"
        # triggers the branch but int() fails, so kwargs stays empty... but
        # retention_days is truthy, so it enters the configure branch with empty kwargs
        service.configure.return_value = {"enabled": False}
        tool.execute(retention_days="abc")
        service.configure.assert_called_once_with()
