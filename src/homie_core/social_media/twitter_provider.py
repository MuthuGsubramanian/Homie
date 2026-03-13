"""Twitter/X provider — feed, profile, publish, and DM via Twitter API v2."""
from __future__ import annotations

import logging
import time

import requests

from homie_core.social_media.models import (
    Conversation,
    DirectMessage,
    ProfileInfo,
    ProfileStats,
    SocialPost,
)
from homie_core.social_media.provider import (
    DirectMessageProvider,
    FeedProvider,
    ProfileProvider,
    PublishProvider,
    SocialMediaProviderBase,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.twitter.com/2"


class TwitterProvider(
    SocialMediaProviderBase,
    FeedProvider,
    ProfileProvider,
    PublishProvider,
    DirectMessageProvider,
):
    """Twitter/X integration using API v2."""

    platform_name: str = "twitter"

    def __init__(self) -> None:
        super().__init__()
        self._user_id: str | None = None
        self._username: str | None = None
        self._refresh_token_str: str | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, credential) -> bool:  # noqa: ANN001
        try:
            self._token = credential.access_token
            self._refresh_token_str = getattr(credential, "refresh_token", None)
            resp = self._call("GET", "/users/me", params={"user.fields": "id,username"})
            data = resp.get("data", {})
            self._user_id = data.get("id")
            self._username = data.get("username")
            self._connected = True
            return True
        except Exception:
            logger.exception("Failed to connect %s", self.platform_name)
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Central API helper
    # ------------------------------------------------------------------

    def _call(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
        retries: int = 2,
    ) -> dict:
        url = f"{BASE_URL}{path}"
        headers = {"Authorization": f"Bearer {self._token}"}

        for attempt in range(retries + 1):
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, params=params, timeout=30)
            else:
                resp = requests.post(
                    url, headers=headers, params=params, json=json_body, timeout=30,
                )

            if resp.status_code == 429 and attempt < retries:
                wait = int(resp.headers.get("Retry-After", "5"))
                logger.warning("Rate-limited by Twitter, retrying in %ss", wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        return {}  # pragma: no cover

    # ------------------------------------------------------------------
    # FeedProvider
    # ------------------------------------------------------------------

    def get_feed(self, limit: int = 20) -> list[SocialPost]:
        try:
            data = self._call(
                "GET",
                f"/users/{self._user_id}/timelines/reverse_chronological",
                params={
                    "max_results": limit,
                    "tweet.fields": "created_at,author_id,public_metrics",
                },
            )
            return [self._tweet_to_post(t) for t in data.get("data", [])]
        except Exception:
            logger.exception("get_feed failed")
            return []

    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]:
        try:
            data = self._call(
                "GET",
                "/tweets/search/recent",
                params={
                    "query": query,
                    "max_results": max(limit, 10),
                    "tweet.fields": "created_at,author_id,public_metrics",
                },
            )
            return [self._tweet_to_post(t) for t in data.get("data", [])]
        except Exception:
            logger.exception("search_posts failed")
            return []

    # ------------------------------------------------------------------
    # ProfileProvider
    # ------------------------------------------------------------------

    def get_profile(self, username: str | None = None) -> ProfileInfo:
        try:
            user_fields = "id,username,name,description,profile_image_url,verified,created_at"
            if username is None:
                data = self._call(
                    "GET", "/users/me", params={"user.fields": user_fields},
                )
            else:
                data = self._call(
                    "GET",
                    f"/users/by/username/{username}",
                    params={"user.fields": user_fields},
                )
            u = data.get("data", {})
            return ProfileInfo(
                platform="twitter",
                username=u.get("username", ""),
                display_name=u.get("name", ""),
                bio=u.get("description", ""),
                avatar_url=u.get("profile_image_url"),
                profile_url=f"https://x.com/{u.get('username', '')}",
                verified=u.get("verified", False),
            )
        except Exception:
            logger.exception("get_profile failed")
            return ProfileInfo(
                platform="twitter", username="", display_name="", bio="",
            )

    def get_stats(self) -> ProfileStats:
        try:
            data = self._call(
                "GET",
                "/users/me",
                params={"user.fields": "public_metrics"},
            )
            metrics = data.get("data", {}).get("public_metrics", {})
            followers = metrics.get("followers_count", 0)
            following = metrics.get("following_count", 0)
            tweets = metrics.get("tweet_count", 0)
            engagement = (
                (metrics.get("listed_count", 0) / followers) if followers else 0.0
            )
            return ProfileStats(
                platform="twitter",
                followers=followers,
                following=following,
                post_count=tweets,
                engagement_rate=engagement,
            )
        except Exception:
            logger.exception("get_stats failed")
            return ProfileStats(platform="twitter")

    # ------------------------------------------------------------------
    # PublishProvider
    # ------------------------------------------------------------------

    def publish(self, content: str, media_urls: list[str] | None = None) -> dict:
        try:
            data = self._call("POST", "/tweets", json_body={"text": content})
            tweet = data.get("data", {})
            return {
                "status": "published",
                "post_id": tweet.get("id", ""),
                "platform": "twitter",
            }
        except Exception:
            logger.exception("publish failed")
            return {"status": "error", "platform": "twitter"}

    # ------------------------------------------------------------------
    # DirectMessageProvider
    # ------------------------------------------------------------------

    def list_conversations(self, limit: int = 20) -> list[Conversation]:
        try:
            data = self._call(
                "GET", "/dm_conversations", params={"max_results": limit},
            )
            convos: list[Conversation] = []
            for c in data.get("data", []):
                convos.append(
                    Conversation(
                        id=c.get("id", ""),
                        platform="twitter",
                        participants=c.get("participants", []),
                        last_message_preview=c.get("last_message", ""),
                        last_activity=c.get("last_activity", 0.0),
                    ),
                )
            return convos
        except Exception:
            logger.exception("list_conversations failed")
            return []

    def get_messages(
        self, conversation_id: str, limit: int = 20,
    ) -> list[DirectMessage]:
        try:
            data = self._call(
                "GET",
                f"/dm_conversations/{conversation_id}/dm_events",
                params={"max_results": limit},
            )
            msgs: list[DirectMessage] = []
            for m in data.get("data", []):
                msgs.append(
                    DirectMessage(
                        id=m.get("id", ""),
                        platform="twitter",
                        conversation_id=conversation_id,
                        sender=m.get("sender_id", ""),
                        content=m.get("text", ""),
                        timestamp=m.get("created_at", 0.0),
                    ),
                )
            return msgs
        except Exception:
            logger.exception("get_messages failed")
            return []

    def send_message(self, recipient: str, text: str) -> dict:
        try:
            self._call(
                "POST",
                f"/dm_conversations/with/{recipient}/messages",
                json_body={"text": text},
            )
            return {"status": "sent", "platform": "twitter", "recipient": recipient}
        except Exception:
            logger.exception("send_message failed")
            return {"status": "error", "platform": "twitter"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tweet_to_post(tweet: dict) -> SocialPost:
        metrics = tweet.get("public_metrics", {})
        return SocialPost(
            id=tweet.get("id", ""),
            platform="twitter",
            author=tweet.get("author_id", ""),
            content=tweet.get("text", ""),
            timestamp=tweet.get("created_at", 0.0),
            url=f"https://x.com/i/status/{tweet.get('id', '')}",
            likes=metrics.get("like_count", 0),
            comments=metrics.get("reply_count", 0),
            shares=metrics.get("retweet_count", 0),
        )
