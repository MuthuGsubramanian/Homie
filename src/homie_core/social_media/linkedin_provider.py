"""LinkedIn social media provider — feed, profile, publish (no DM)."""
from __future__ import annotations

import logging
import time

import requests

from homie_core.social_media.models import ProfileInfo, ProfileStats, SocialPost
from homie_core.social_media.provider import (
    FeedProvider,
    ProfileProvider,
    PublishProvider,
    SocialMediaProviderBase,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.linkedin.com/v2"


class LinkedInProvider(SocialMediaProviderBase, FeedProvider, ProfileProvider, PublishProvider):
    """LinkedIn REST API provider.

    Implements Feed, Profile, and Publish capabilities.
    Does **not** implement DirectMessageProvider — the LinkedIn API does not
    expose messaging endpoints to third-party applications.
    """

    platform_name: str = "linkedin"

    def __init__(self) -> None:
        super().__init__()
        self._person_id: str | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, credential) -> bool:
        """Store token, call ``/me`` to verify, and cache person URN."""
        try:
            self._token = credential.access_token
            resp = self._call("GET", "/me")
            self._person_id = resp.get("id")
            self._connected = True
            return True
        except Exception:
            logger.exception("Failed to connect LinkedIn")
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    def _call(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
        retries: int = 2,
    ) -> dict:
        """Issue an authenticated request to the LinkedIn API.

        Automatically retries on 429 (rate-limit) responses up to *retries*
        times with a 1-second back-off.
        """
        url = f"{BASE_URL}{path}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-Restli-Protocol-Version": "2.0.0",
        }

        for attempt in range(retries + 1):
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
            )

            if resp.status_code == 429 and attempt < retries:
                time.sleep(1)
                continue

            resp.raise_for_status()
            return resp.json()

        # Should not reach here, but satisfy type-checkers.
        resp.raise_for_status()  # type: ignore[possibly-undefined]
        return resp.json()  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------
    # FeedProvider
    # ------------------------------------------------------------------

    def get_feed(self, limit: int = 20) -> list[SocialPost]:
        data = self._call("GET", "/feed", params={"count": limit})
        posts: list[SocialPost] = []
        for item in data.get("elements", []):
            posts.append(
                SocialPost(
                    id=item.get("id", ""),
                    platform="linkedin",
                    author=item.get("author", ""),
                    content=item.get("text", ""),
                    timestamp=item.get("created", {}).get("time", 0.0),
                    likes=item.get("likes", 0),
                    comments=item.get("comments", 0),
                    shares=item.get("shares", 0),
                )
            )
        return posts

    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]:
        data = self._call(
            "GET",
            "/search/blended",
            params={"q": "content", "keywords": query, "count": limit},
        )
        posts: list[SocialPost] = []
        for item in data.get("elements", []):
            posts.append(
                SocialPost(
                    id=item.get("id", ""),
                    platform="linkedin",
                    author=item.get("author", ""),
                    content=item.get("text", ""),
                    timestamp=item.get("created", {}).get("time", 0.0),
                )
            )
        return posts

    # ------------------------------------------------------------------
    # ProfileProvider
    # ------------------------------------------------------------------

    def get_profile(self, username: str | None = None) -> ProfileInfo:
        data = self._call(
            "GET",
            "/me",
            params={
                "projection": "(id,firstName,lastName,headline,profilePicture,vanityName)"
            },
        )

        first = _localized_field(data.get("firstName", {}))
        last = _localized_field(data.get("lastName", {}))
        display_name = f"{first} {last}".strip()

        return ProfileInfo(
            platform="linkedin",
            username=data.get("vanityName", data.get("id", "")),
            display_name=display_name,
            bio=data.get("headline", {}).get("localized", {}).get("en_US", "")
            if isinstance(data.get("headline"), dict)
            else str(data.get("headline", "")),
            avatar_url=data.get("profilePicture", {})
            .get("displayImage~", {})
            .get("elements", [{}])[0]
            .get("identifiers", [{}])[0]
            .get("identifier"),
            profile_url=f"https://www.linkedin.com/in/{data.get('vanityName', '')}",
        )

    def get_stats(self) -> ProfileStats:
        data = self._call("GET", "/me", params={"projection": "(numConnections)"})
        return ProfileStats(
            platform="linkedin",
            followers=data.get("numConnections", 0),
        )

    # ------------------------------------------------------------------
    # PublishProvider
    # ------------------------------------------------------------------

    def publish(self, content: str, media_urls: list[str] | None = None) -> dict:
        person_urn = f"urn:li:person:{self._person_id}"
        payload: dict = {
            "author": person_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": content},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }

        if media_urls:
            media_items = [
                {"status": "READY", "originalUrl": url} for url in media_urls
            ]
            share = payload["specificContent"]["com.linkedin.ugc.ShareContent"]
            share["shareMediaCategory"] = "ARTICLE"
            share["media"] = media_items

        data = self._call("POST", "/ugcPosts", json_body=payload)
        return {"id": data.get("id", ""), "status": "published"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _localized_field(field_obj: dict) -> str:
    """Extract the first localized value from a LinkedIn multi-locale field."""
    localized = field_obj.get("localized", {})
    if not localized:
        return ""
    # Return the first available locale value.
    return next(iter(localized.values()), "")
