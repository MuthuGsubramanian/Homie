"""Tests for browser history data models."""
import pytest
from homie_core.browser.models import HistoryEntry, BrowsingPattern, BrowserConfig


class TestHistoryEntry:
    def test_to_dict(self):
        entry = HistoryEntry(
            url="https://example.com/page",
            title="Example Page",
            visit_time=1700000000.0,
            duration=5.5,
            browser="chrome",
        )
        d = entry.to_dict()
        assert d["url"] == "https://example.com/page"
        assert d["title"] == "Example Page"
        assert d["visit_time"] == 1700000000.0
        assert d["duration"] == 5.5
        assert d["browser"] == "chrome"

    def test_to_dict_defaults(self):
        entry = HistoryEntry(url="https://x.com", title="X", visit_time=0)
        d = entry.to_dict()
        assert d["duration"] is None
        assert d["browser"] == "chrome"

    def test_fields_assignable(self):
        entry = HistoryEntry(url="a", title="b", visit_time=1.0, browser="firefox")
        entry.browser = "edge"
        assert entry.browser == "edge"


class TestBrowsingPattern:
    def test_to_dict_empty(self):
        pattern = BrowsingPattern()
        d = pattern.to_dict()
        assert d["top_domains"] == []
        assert d["top_topics"] == []
        assert d["peak_hours"] == []
        assert d["daily_avg_pages"] == 0.0
        assert d["category_breakdown"] == {}

    def test_to_dict_with_data(self):
        pattern = BrowsingPattern(
            top_domains=[{"domain": "github.com", "visit_count": 10}],
            top_topics=["programming"],
            peak_hours=[10, 14],
            daily_avg_pages=25.3,
            category_breakdown={"dev_docs": 60.0, "other": 40.0},
        )
        d = pattern.to_dict()
        assert len(d["top_domains"]) == 1
        assert d["top_domains"][0]["domain"] == "github.com"
        assert d["peak_hours"] == [10, 14]
        assert d["daily_avg_pages"] == 25.3


class TestBrowserConfig:
    def test_defaults(self):
        config = BrowserConfig()
        assert config.enabled is False
        assert config.browsers == ["chrome"]
        assert config.extension_enabled is False
        assert config.exclude_domains == []
        assert config.include_only_domains == []
        assert config.retention_days == 30
        assert config.analyze_urls is True

    def test_custom_values(self):
        config = BrowserConfig(
            enabled=True,
            browsers=["chrome", "firefox"],
            retention_days=7,
        )
        assert config.enabled is True
        assert config.browsers == ["chrome", "firefox"]
        assert config.retention_days == 7
