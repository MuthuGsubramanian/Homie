"""Tests for the Twitter/X social media provider."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from homie_core.social_media.models import ProfileInfo, SocialPost
from homie_core.social_media.twitter_provider import TwitterProvider


# ── helpers ──────────────────────────────────────────────────────────


def _make_provider(mock_requests: MagicMock) -> TwitterProvider:
    """Return a connected TwitterProvider with mocked /users/me."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {"id": "123", "username": "testuser"},
    }
    mock_requests.get.return_value = resp

    provider = TwitterProvider()
    cred = SimpleNamespace(access_token="tok123")
    provider.connect(cred)
    return provider


# ── TestTwitterConnect ───────────────────────────────────────────────


class TestTwitterConnect:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_connect_success(self, mock_requests: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": {"id": "42", "username": "alice"},
        }
        mock_requests.get.return_value = resp

        provider = TwitterProvider()
        cred = SimpleNamespace(access_token="token_abc")
        assert provider.connect(cred) is True
        assert provider.is_connected
        assert provider._user_id == "42"
        assert provider._username == "alice"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_connect_failure(self, mock_requests: MagicMock) -> None:
        mock_requests.get.side_effect = Exception("network error")

        provider = TwitterProvider()
        cred = SimpleNamespace(access_token="bad_token")
        assert provider.connect(cred) is False
        assert not provider.is_connected


# ── TestTwitterFeed ──────────────────────────────────────────────────


class TestTwitterFeed:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_feed(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        feed_resp = MagicMock()
        feed_resp.status_code = 200
        feed_resp.json.return_value = {
            "data": [
                {
                    "id": "t1",
                    "text": "Hello world",
                    "author_id": "123",
                    "created_at": 1700000000.0,
                    "public_metrics": {
                        "like_count": 10,
                        "reply_count": 2,
                        "retweet_count": 5,
                    },
                },
            ],
        }
        mock_requests.get.return_value = feed_resp

        posts = provider.get_feed(limit=5)
        assert len(posts) == 1
        assert isinstance(posts[0], SocialPost)
        assert posts[0].id == "t1"
        assert posts[0].likes == 10
        assert posts[0].platform == "twitter"


# ── TestTwitterProfile ───────────────────────────────────────────────


class TestTwitterProfile:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_profile(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        profile_resp = MagicMock()
        profile_resp.status_code = 200
        profile_resp.json.return_value = {
            "data": {
                "id": "42",
                "username": "alice",
                "name": "Alice",
                "description": "Builder",
                "profile_image_url": "https://img.example.com/a.jpg",
                "verified": True,
            },
        }
        mock_requests.get.return_value = profile_resp

        info = provider.get_profile("alice")
        assert isinstance(info, ProfileInfo)
        assert info.username == "alice"
        assert info.verified is True
        assert info.display_name == "Alice"


# ── TestTwitterPublish ───────────────────────────────────────────────


class TestTwitterPublish:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_publish(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        pub_resp = MagicMock()
        pub_resp.status_code = 201
        pub_resp.json.return_value = {
            "data": {"id": "new_tweet_1"},
        }
        mock_requests.post.return_value = pub_resp

        result = provider.publish("Check this out!")
        assert result["status"] == "published"
        assert result["post_id"] == "new_tweet_1"


# ── TestTwitterDM ────────────────────────────────────────────────────


class TestTwitterDM:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_send_message(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        dm_resp = MagicMock()
        dm_resp.status_code = 200
        dm_resp.json.return_value = {"data": {"id": "dm1"}}
        mock_requests.post.return_value = dm_resp

        result = provider.send_message("bob", "hey!")
        assert result["status"] == "sent"
        assert result["recipient"] == "bob"


# ── TestTwitterRateLimit ─────────────────────────────────────────────


class TestTwitterRateLimit:
    @patch("homie_core.social_media.twitter_provider.time.sleep")
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_rate_limit_retry(
        self, mock_requests: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "3"}

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {
            "data": {"id": "123", "username": "testuser"},
        }

        mock_requests.get.side_effect = [rate_resp, ok_resp]

        provider = TwitterProvider()
        provider._token = "tok"
        result = provider._call("GET", "/users/me")

        assert result == {"data": {"id": "123", "username": "testuser"}}
        mock_sleep.assert_called_once_with(3)
