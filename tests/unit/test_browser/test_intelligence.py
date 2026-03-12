"""Tests for browsing pattern analysis."""
import pytest
from homie_core.browser.models import HistoryEntry, BrowsingPattern
from homie_core.browser.intelligence import BrowsingIntelligence


class TestBrowsingIntelligence:
    def _make_entry(self, url, title="", visit_time=1700000000.0, browser="chrome"):
        return HistoryEntry(url=url, title=title, visit_time=visit_time, browser=browser)

    def test_analyze_empty(self):
        intel = BrowsingIntelligence()
        pattern = intel.analyze([])
        assert pattern.top_domains == []
        assert pattern.daily_avg_pages == 0.0
        assert pattern.category_breakdown == {}

    def test_analyze_single_entry(self):
        intel = BrowsingIntelligence()
        entries = [self._make_entry("https://github.com/repo", "Repo")]
        pattern = intel.analyze(entries)
        assert len(pattern.top_domains) == 1
        assert pattern.top_domains[0]["domain"] == "github.com"
        assert pattern.top_domains[0]["visit_count"] == 1
        assert "dev_docs" in pattern.category_breakdown

    def test_analyze_multiple_entries(self):
        intel = BrowsingIntelligence()
        entries = [
            self._make_entry("https://github.com/repo", "Repo", 1700000000.0),
            self._make_entry("https://github.com/other", "Other", 1700001000.0),
            self._make_entry("https://news.bbc.co.uk/article", "BBC News", 1700002000.0),
        ]
        pattern = intel.analyze(entries)

        # github.com should be top domain with 2 visits
        domains = {d["domain"]: d["visit_count"] for d in pattern.top_domains}
        assert domains.get("github.com") == 2
        assert domains.get("news.bbc.co.uk") == 1

        # Categories
        assert "dev_docs" in pattern.category_breakdown
        assert "news" in pattern.category_breakdown

    def test_analyze_categories(self):
        intel = BrowsingIntelligence()
        entries = [
            self._make_entry("https://youtube.com/watch?v=123"),
            self._make_entry("https://amazon.com/product"),
            self._make_entry("https://reddit.com/r/python"),
            self._make_entry("https://someunknown.com/page"),
        ]
        pattern = intel.analyze(entries)
        assert "video" in pattern.category_breakdown
        assert "shopping" in pattern.category_breakdown
        assert "social" in pattern.category_breakdown
        assert "other" in pattern.category_breakdown

    def test_analyze_peak_hours(self):
        intel = BrowsingIntelligence()
        # Create entries at different times
        entries = [
            self._make_entry("https://a.com", visit_time=1700000000.0),  # some hour
            self._make_entry("https://b.com", visit_time=1700000100.0),  # same hour
            self._make_entry("https://c.com", visit_time=1700050000.0),  # different hour
        ]
        pattern = intel.analyze(entries)
        assert len(pattern.peak_hours) > 0

    def test_analyze_daily_avg(self):
        intel = BrowsingIntelligence()
        # Entries spanning 2 days (172800 seconds)
        entries = [
            self._make_entry("https://a.com", visit_time=1700000000.0),
            self._make_entry("https://b.com", visit_time=1700086400.0),  # +1 day
            self._make_entry("https://c.com", visit_time=1700172800.0),  # +2 days
        ]
        pattern = intel.analyze(entries)
        assert pattern.daily_avg_pages > 0
        # 3 entries over 2 days = 1.5 per day
        assert abs(pattern.daily_avg_pages - 1.5) < 0.1

    def test_analyze_returns_browsing_pattern(self):
        intel = BrowsingIntelligence()
        entries = [self._make_entry("https://example.com")]
        result = intel.analyze(entries)
        assert isinstance(result, BrowsingPattern)

    def test_category_breakdown_sums_to_100(self):
        intel = BrowsingIntelligence()
        entries = [
            self._make_entry("https://github.com/a"),
            self._make_entry("https://youtube.com/b"),
            self._make_entry("https://amazon.com/c"),
            self._make_entry("https://unknown.com/d"),
        ]
        pattern = intel.analyze(entries)
        total = sum(pattern.category_breakdown.values())
        assert abs(total - 100.0) < 0.5
