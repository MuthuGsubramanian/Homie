"""Tests for weather/news/briefing brain tools."""
import pytest
from unittest.mock import MagicMock, patch
from homie_core.brain.tool_registry import ToolRegistry


def test_intelligence_tools_registered():
    from homie_core.brain.builtin_tools import register_intelligence_tools
    registry = ToolRegistry()
    register_intelligence_tools(registry, config=MagicMock(), vault=MagicMock())
    tool_names = [t.name for t in registry.list_tools()]
    assert "get_weather" in tool_names
    assert "get_news" in tool_names


def test_weather_tool_uses_config_location():
    from homie_core.brain.builtin_tools import register_intelligence_tools
    registry = ToolRegistry()
    cfg = MagicMock()
    cfg.location = MagicMock(city="Chennai", country="IN")

    vault = MagicMock()
    vault.get_credential.return_value = MagicMock(access_token="test_key")

    register_intelligence_tools(registry, config=cfg, vault=vault)

    tool = registry.get("get_weather")
    assert tool is not None

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "main": {"temp": 32, "humidity": 65, "feels_like": 35},
        "weather": [{"description": "clear sky"}],
        "wind": {"speed": 3.5},
        "name": "Chennai",
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        from homie_core.brain.tool_registry import ToolCall
        result = registry.execute(ToolCall(name="get_weather", arguments={}))
    assert result.success
    assert "Chennai" in result.output


def test_news_tool_returns_headlines():
    from homie_core.brain.builtin_tools import register_intelligence_tools
    registry = ToolRegistry()
    cfg = MagicMock()
    cfg.location = MagicMock(city="Chennai", country="IN")

    vault = MagicMock()
    vault.get_credential.return_value = MagicMock(access_token="test_key")

    register_intelligence_tools(registry, config=cfg, vault=vault)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "ok",
        "articles": [
            {"title": "Test headline", "source": {"name": "Test"}, "url": "http://example.com", "description": "desc"},
        ],
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        from homie_core.brain.tool_registry import ToolCall
        result = registry.execute(ToolCall(name="get_news", arguments={}))
    assert result.success
    assert "Test headline" in result.output


def test_weather_tool_no_location():
    from homie_core.brain.builtin_tools import register_intelligence_tools
    registry = ToolRegistry()
    cfg = MagicMock()
    cfg.location = None

    register_intelligence_tools(registry, config=cfg, vault=MagicMock())

    from homie_core.brain.tool_registry import ToolCall
    result = registry.execute(ToolCall(name="get_weather", arguments={}))
    assert result.success
    assert "No location" in result.output
