"""Cross-platform social intelligence analysis."""
from __future__ import annotations
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from homie_core.social_media.provider import (
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
)

logger = logging.getLogger(__name__)


@dataclass
class PlatformProfile:
    topics: list[str] = field(default_factory=list)
    tone: str = "neutral"
    avg_posts_per_week: float = 0.0
    peak_hours: list[int] = field(default_factory=list)
    audience_size: int = 0
    engagement_rate: float = 0.0
    content_types: dict[str, float] = field(default_factory=dict)


@dataclass
class CrossPlatformAnalysis:
    primary_topics: list[str] = field(default_factory=list)
    peak_hours: list[int] = field(default_factory=list)
    platform_preferences: dict[str, str] = field(default_factory=dict)
    audience_overlap: dict = field(default_factory=dict)
    recommended_posting_times: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class SocialProfile:
    platforms: dict[str, PlatformProfile] = field(default_factory=dict)
    cross_platform: CrossPlatformAnalysis = field(default_factory=CrossPlatformAnalysis)


def _extract_topics(texts: list[str]) -> list[str]:
    """Extract topics from post texts using hashtags and keyword frequency."""
    words = Counter()
    stop_words = {"this", "that", "with", "from", "have", "been", "just", "about",
                  "what", "when", "they", "will", "your", "more", "some", "like",
                  "them", "than", "then", "also", "into", "only", "very", "much"}
    for text in texts:
        hashtags = re.findall(r"#(\w+)", text.lower())
        words.update(hashtags)
        for word in re.findall(r"\b[a-z]{4,}\b", text.lower()):
            if word not in stop_words:
                words[word] += 1
    return [w for w, _ in words.most_common(10)]


class SocialIntelligence:
    """Analyzes connected social media profiles to build behavioral model."""

    def __init__(self, vault):
        self._vault = vault

    def analyze_profiles(self, providers: dict[str, SocialMediaProviderBase]) -> SocialProfile:
        """Fetch profile data from all providers, analyze patterns, store in vault."""
        profile = SocialProfile()
        all_topics: list[str] = []

        for name, provider in providers.items():
            plat = PlatformProfile()

            if isinstance(provider, ProfileProvider):
                try:
                    stats = provider.get_stats()
                    plat.audience_size = stats.followers
                    plat.engagement_rate = stats.engagement_rate
                except Exception:
                    logger.exception("Failed to get stats for %s", name)

            if isinstance(provider, FeedProvider):
                try:
                    posts = provider.get_feed(limit=50)
                    texts = [p.content for p in posts]
                    plat.topics = _extract_topics(texts)
                    all_topics.extend(plat.topics)

                    # Content type breakdown
                    type_counts: dict[str, int] = {}
                    for p in posts:
                        type_counts[p.post_type] = type_counts.get(p.post_type, 0) + 1
                    total = len(posts) or 1
                    plat.content_types = {k: round(v / total, 2) for k, v in type_counts.items()}
                except Exception:
                    logger.exception("Failed to analyze feed for %s", name)

            profile.platforms[name] = plat

        # Cross-platform analysis
        profile.cross_platform.primary_topics = [
            t for t, _ in Counter(all_topics).most_common(10)
        ]

        self._store(profile)
        return profile

    def get_cached_profile(self) -> SocialProfile | None:
        """Load previously stored analysis from vault."""
        cred = self._vault.get_credential("social_intelligence", "profile")
        if not cred:
            return None
        try:
            data = json.loads(cred.access_token)
            profile = SocialProfile()
            profile.cross_platform = CrossPlatformAnalysis(
                primary_topics=data.get("cross_platform", {}).get("primary_topics", []),
            )
            for name, pdata in data.get("platforms", {}).items():
                profile.platforms[name] = PlatformProfile(
                    topics=pdata.get("topics", []),
                    tone=pdata.get("tone", "neutral"),
                    audience_size=pdata.get("audience_size", 0),
                    engagement_rate=pdata.get("engagement_rate", 0.0),
                )
            return profile
        except Exception:
            logger.exception("Failed to load cached social profile")
            return None

    def _store(self, profile: SocialProfile) -> None:
        data = {
            "last_scan": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platforms": {
                name: {
                    "topics": p.topics, "tone": p.tone,
                    "audience_size": p.audience_size,
                    "engagement_rate": p.engagement_rate,
                    "content_types": p.content_types,
                }
                for name, p in profile.platforms.items()
            },
            "cross_platform": {
                "primary_topics": profile.cross_platform.primary_topics,
            },
        }
        try:
            self._vault.store_credential(
                provider="social_intelligence", account_id="profile",
                token_type="data", access_token=json.dumps(data),
                refresh_token="", scopes=[],
            )
        except Exception:
            logger.exception("Failed to store social intelligence profile")
