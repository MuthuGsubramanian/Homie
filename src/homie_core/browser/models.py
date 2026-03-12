"""Data models for browser history."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class HistoryEntry:
    url: str
    title: str
    visit_time: float
    duration: float | None = None
    browser: str = "chrome"

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "title": self.title, "visit_time": self.visit_time,
                "duration": self.duration, "browser": self.browser}

@dataclass
class BrowsingPattern:
    top_domains: list[dict] = field(default_factory=list)
    top_topics: list[str] = field(default_factory=list)
    peak_hours: list[int] = field(default_factory=list)
    daily_avg_pages: float = 0.0
    category_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"top_domains": self.top_domains, "top_topics": self.top_topics,
                "peak_hours": self.peak_hours, "daily_avg_pages": self.daily_avg_pages,
                "category_breakdown": self.category_breakdown}

@dataclass
class BrowserConfig:
    enabled: bool = False
    browsers: list[str] = field(default_factory=lambda: ["chrome"])
    extension_enabled: bool = False
    exclude_domains: list[str] = field(default_factory=list)
    include_only_domains: list[str] = field(default_factory=list)
    retention_days: int = 30
    analyze_urls: bool = True
