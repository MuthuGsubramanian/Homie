"""Browsing pattern analysis."""
from __future__ import annotations
import time
from collections import Counter
from urllib.parse import urlparse
from homie_core.browser.models import HistoryEntry, BrowsingPattern

_CATEGORY_KEYWORDS = {
    "news": ["news", "bbc", "cnn", "reuters", "nytimes"],
    "dev_docs": ["docs", "documentation", "stackoverflow", "github", "gitlab", "mdn"],
    "social": ["twitter", "reddit", "facebook", "instagram", "linkedin", "x.com"],
    "video": ["youtube", "vimeo", "twitch", "netflix"],
    "shopping": ["amazon", "ebay", "shop", "store", "buy"],
    "email": ["gmail", "outlook", "mail"],
}

class BrowsingIntelligence:
    def analyze(self, entries: list[HistoryEntry]) -> BrowsingPattern:
        if not entries:
            return BrowsingPattern()

        domain_counts: Counter = Counter()
        hour_counts: Counter = Counter()
        categories: Counter = Counter()

        for entry in entries:
            try:
                domain = urlparse(entry.url).netloc
            except Exception:
                domain = ""
            domain_counts[domain] += 1

            t = time.localtime(entry.visit_time) if entry.visit_time else None
            if t:
                hour_counts[t.tm_hour] += 1

            # Categorize
            url_lower = entry.url.lower()
            categorized = False
            for cat, keywords in _CATEGORY_KEYWORDS.items():
                if any(kw in url_lower for kw in keywords):
                    categories[cat] += 1
                    categorized = True
                    break
            if not categorized:
                categories["other"] += 1

        total = len(entries)
        # Estimate daily average (assume entries span some days)
        if entries:
            time_span = max(e.visit_time for e in entries) - min(e.visit_time for e in entries)
            days = max(time_span / 86400, 1)
            daily_avg = total / days
        else:
            daily_avg = 0

        return BrowsingPattern(
            top_domains=[{"domain": d, "visit_count": c} for d, c in domain_counts.most_common(10)],
            peak_hours=[h for h, _ in hour_counts.most_common(5)],
            daily_avg_pages=round(daily_avg, 1),
            category_breakdown={cat: round(count / total * 100, 1) for cat, count in categories.most_common()},
        )
