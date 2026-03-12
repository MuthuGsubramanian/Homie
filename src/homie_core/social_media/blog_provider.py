"""Blog provider — RSS/Atom feed reader (read-only)."""
from __future__ import annotations
import calendar
import logging
import time

import feedparser

from homie_core.social_media.models import SocialPost, ProfileInfo, ProfileStats
from homie_core.social_media.provider import (
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
)

logger = logging.getLogger(__name__)


class BlogProvider(SocialMediaProviderBase, FeedProvider, ProfileProvider):
    platform_name = "blog"

    def __init__(self):
        super().__init__()
        self._feed_url: str | None = None
        self._feed_data: dict | None = None

    def connect(self, credential) -> bool:
        try:
            self._feed_url = credential.access_token  # URL stored as token
            self._feed_data = feedparser.parse(self._feed_url)
            self._connected = bool(self._feed_data.get("entries"))
            return self._connected
        except Exception:
            logger.exception("Blog connect failed")
            return False

    def _fetch(self) -> dict:
        if self._feed_url:
            self._feed_data = feedparser.parse(self._feed_url)
        return self._feed_data or {}

    def get_feed(self, limit: int = 20) -> list[SocialPost]:
        data = self._fetch()
        posts = []
        for entry in data.get("entries", [])[:limit]:
            ts = 0.0
            if entry.get("published_parsed"):
                ts = calendar.timegm(entry["published_parsed"])
            posts.append(SocialPost(
                id=entry.get("id", entry.get("link", "")),
                platform="blog", author=entry.get("author", ""),
                content=entry.get("summary", entry.get("title", "")),
                timestamp=ts, url=entry.get("link"),
                post_type="article",
            ))
        return posts

    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]:
        q = query.lower()
        return [p for p in self.get_feed(limit=100)
                if q in p.content.lower() or q in (p.url or "").lower()][:limit]

    def get_profile(self, username: str | None = None) -> ProfileInfo:
        data = self._fetch()
        feed = data.get("feed", {})
        return ProfileInfo(
            platform="blog", username=self._feed_url or "",
            display_name=feed.get("title", ""),
            bio=feed.get("subtitle", feed.get("description", "")),
            profile_url=feed.get("link", self._feed_url),
        )

    def get_stats(self) -> ProfileStats:
        data = self._fetch()
        return ProfileStats(
            platform="blog", post_count=len(data.get("entries", [])),
        )
