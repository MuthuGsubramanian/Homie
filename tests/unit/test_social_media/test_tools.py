"""Tests for social media AI tools."""
import json
from unittest.mock import MagicMock
from homie_core.brain.tool_registry import ToolRegistry
from homie_core.social_media.tools import register_social_media_tools


class TestToolRegistration:
    def test_registers_9_tools(self):
        registry = ToolRegistry()
        service = MagicMock()
        register_social_media_tools(registry, service)
        names = {t.name for t in registry.list_tools()}
        expected = {"sm_feed", "sm_profile", "sm_scan_profiles", "sm_publish",
                    "sm_conversations", "sm_dms", "sm_send_dm", "sm_search", "sm_notifications"}
        assert expected.issubset(names)


class TestSmFeedTool:
    def test_returns_json(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_feed.return_value = [{"id": "1", "content": "Hello"}]
        register_social_media_tools(registry, service)
        result = registry.get("sm_feed").execute()
        data = json.loads(result)
        assert len(data) == 1

    def test_invalid_limit(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_feed.return_value = []
        register_social_media_tools(registry, service)
        result = registry.get("sm_feed").execute(limit="abc")
        assert json.loads(result) == []


class TestSmProfileTool:
    def test_requires_platform(self):
        registry = ToolRegistry()
        service = MagicMock()
        register_social_media_tools(registry, service)
        result = registry.get("sm_profile").execute()
        assert "error" in json.loads(result)

    def test_returns_profile(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_profile.return_value = {"username": "@test"}
        register_social_media_tools(registry, service)
        result = registry.get("sm_profile").execute(platform="twitter")
        assert json.loads(result)["username"] == "@test"


class TestSmPublishTool:
    def test_requires_fields(self):
        registry = ToolRegistry()
        service = MagicMock()
        register_social_media_tools(registry, service)
        result = registry.get("sm_publish").execute()
        assert "error" in json.loads(result)

    def test_publish_success(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.publish.return_value = {"status": "published"}
        register_social_media_tools(registry, service)
        result = registry.get("sm_publish").execute(platform="twitter", content="Hello!")
        assert json.loads(result)["status"] == "published"


class TestSmSearchTool:
    def test_requires_query(self):
        registry = ToolRegistry()
        service = MagicMock()
        register_social_media_tools(registry, service)
        result = registry.get("sm_search").execute()
        assert "error" in json.loads(result)

    def test_search(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.search.return_value = [{"content": "match"}]
        register_social_media_tools(registry, service)
        result = registry.get("sm_search").execute(query="test")
        assert len(json.loads(result)) == 1


class TestSmSendDmTool:
    def test_requires_all_fields(self):
        registry = ToolRegistry()
        service = MagicMock()
        register_social_media_tools(registry, service)
        result = registry.get("sm_send_dm").execute()
        assert "error" in json.loads(result)
