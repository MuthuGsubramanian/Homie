"""Tests for the Reddit social-media provider."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests as _real_requests

from homie_core.social_media.models import (
    Conversation,
    DirectMessage,
    ProfileInfo,
    ProfileStats,
    SocialPost,
)
from homie_core.social_media.reddit_provider import RedditProvider


# -- helpers ----------------------------------------------------------------


def _make_credential(token: str = "tok123") -> MagicMock:
    cred = MagicMock()
    cred.access_token = token
    return cred


def _me_response() -> dict:
    return {
        "name": "test_user",
        "total_karma": 4200,
        "created_utc": 1600000000.0,
        "icon_img": "https://example.com/avatar.png",
        "verified": True,
        "subreddit": {
            "title": "Test User",
            "public_description": "Hello, I am a test user.",
        },
    }


def _listing(children: list[dict]) -> dict:
    return {"data": {"children": [{"data": c} for c in children]}}


def _make_provider(mock_requests: MagicMock) -> RedditProvider:
    """Return a connected RedditProvider."""
    connect_resp = MagicMock(status_code=200)
    connect_resp.json.return_value = _me_response()
    mock_requests.request.return_value = connect_resp

    provider = RedditProvider()
    provider.connect(_make_credential())
    return provider


SAMPLE_POST = {
    "name": "t3_abc123",
    "author": "poster",
    "title": "Cool post",
    "selftext": "Some body text",
    "created_utc": 1700000000.0,
    "permalink": "/r/test/comments/abc123/cool_post/",
    "ups": 42,
    "num_comments": 7,
    "is_self": True,
}

SAMPLE_LINK_POST = {
    "name": "t3_link1",
    "author": "linker",
    "title": "External link",
    "selftext": "",
    "created_utc": 1700000001.0,
    "permalink": "/r/news/comments/link1/external_link/",
    "ups": 100,
    "num_comments": 20,
    "is_self": False,
}


# -- TestConnect ------------------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestConnect:
    def test_connect_success(self, mock_requests):
        resp = MagicMock(status_code=200)
        resp.json.return_value = _me_response()
        mock_requests.request.return_value = resp

        provider = RedditProvider()
        assert provider.connect(_make_credential()) is True
        assert provider.is_connected is True
        assert provider._username == "test_user"

    def test_connect_failure(self, mock_requests):
        resp = MagicMock(status_code=401)
        resp.raise_for_status.side_effect = Exception("Unauthorized")
        resp.json.side_effect = Exception("Unauthorized")
        mock_requests.request.return_value = resp

        provider = RedditProvider()
        assert provider.connect(_make_credential()) is False
        assert provider.is_connected is False

    def test_connect_stores_token(self, mock_requests):
        resp = MagicMock(status_code=200)
        resp.json.return_value = _me_response()
        mock_requests.request.return_value = resp

        provider = RedditProvider()
        provider.connect(_make_credential("my_secret_token"))
        assert provider._token == "my_secret_token"


# -- TestFeed ---------------------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestFeed:
    def test_get_feed(self, mock_requests):
        connect_resp = MagicMock(status_code=200)
        connect_resp.json.return_value = _me_response()

        feed_resp = MagicMock(status_code=200)
        feed_resp.json.return_value = _listing([SAMPLE_POST])

        mock_requests.request.side_effect = [connect_resp, feed_resp]

        provider = RedditProvider()
        provider.connect(_make_credential())
        posts = provider.get_feed(limit=5)

        assert len(posts) == 1
        post = posts[0]
        assert post.id == "t3_abc123"
        assert post.author == "poster"
        assert post.content == "Some body text"
        assert post.likes == 42
        assert post.comments == 7
        assert post.platform == "reddit"

    def test_get_feed_empty(self, mock_requests):
        provider = _make_provider(mock_requests)

        empty_resp = MagicMock(status_code=200)
        empty_resp.json.return_value = _listing([])
        mock_requests.request.return_value = empty_resp

        assert provider.get_feed(limit=5) == []

    def test_search_posts(self, mock_requests):
        provider = _make_provider(mock_requests)

        search_resp = MagicMock(status_code=200)
        search_resp.json.return_value = _listing([SAMPLE_POST])
        mock_requests.request.return_value = search_resp

        posts = provider.search_posts("cool", limit=10)
        assert len(posts) == 1
        assert posts[0].content == "Some body text"


# -- TestProfile ------------------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestProfile:
    def test_get_profile_self(self, mock_requests):
        me = _me_response()
        resp = MagicMock(status_code=200)
        resp.json.return_value = me
        mock_requests.request.return_value = resp

        provider = RedditProvider()
        provider.connect(_make_credential())
        profile = provider.get_profile()

        assert profile.username == "test_user"
        assert profile.display_name == "Test User"
        assert profile.bio == "Hello, I am a test user."
        assert profile.platform == "reddit"
        assert profile.verified is True
        assert profile.profile_url == "https://www.reddit.com/user/test_user"

    def test_get_profile_other_user(self, mock_requests):
        provider = _make_provider(mock_requests)

        other_resp = MagicMock(status_code=200)
        other_resp.json.return_value = {
            "data": {
                "name": "other_user",
                "verified": False,
                "icon_img": None,
                "subreddit": {
                    "title": "Other",
                    "public_description": "other bio",
                },
            },
        }
        mock_requests.request.return_value = other_resp

        profile = provider.get_profile("other_user")
        assert profile.username == "other_user"
        assert profile.display_name == "Other"

    def test_get_stats(self, mock_requests):
        me = _me_response()
        resp = MagicMock(status_code=200)
        resp.json.return_value = me
        mock_requests.request.return_value = resp

        provider = RedditProvider()
        provider.connect(_make_credential())
        stats = provider.get_stats()

        assert isinstance(stats, ProfileStats)
        assert stats.followers == 4200  # total_karma
        assert stats.platform == "reddit"


# -- TestPublish ------------------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestPublish:
    def test_publish(self, mock_requests):
        connect_resp = MagicMock(status_code=200)
        connect_resp.json.return_value = _me_response()

        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"success": True}

        mock_requests.request.side_effect = [connect_resp, submit_resp]

        provider = RedditProvider()
        provider.connect(_make_credential())
        result = provider.publish("Hello Reddit!")

        assert result == {"success": True}
        submit_call = mock_requests.request.call_args_list[1]
        assert submit_call[0] == ("POST", "https://oauth.reddit.com/api/submit")
        assert submit_call[1]["json"]["kind"] == "self"
        assert submit_call[1]["json"]["sr"] == "u_test_user"
        assert submit_call[1]["json"]["text"] == "Hello Reddit!"


# -- TestDirectMessages -----------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestDirectMessages:
    def test_send_message(self, mock_requests):
        connect_resp = MagicMock(status_code=200)
        connect_resp.json.return_value = _me_response()

        compose_resp = MagicMock(status_code=200)
        compose_resp.json.return_value = {"success": True}

        mock_requests.request.side_effect = [connect_resp, compose_resp]

        provider = RedditProvider()
        provider.connect(_make_credential())
        result = provider.send_message("other_user", "Hey!")

        assert result == {"success": True}
        compose_call = mock_requests.request.call_args_list[1]
        assert compose_call[0] == ("POST", "https://oauth.reddit.com/api/compose")
        assert compose_call[1]["json"]["to"] == "other_user"
        assert compose_call[1]["json"]["text"] == "Hey!"

    def test_list_conversations(self, mock_requests):
        provider = _make_provider(mock_requests)

        inbox_resp = MagicMock(status_code=200)
        inbox_resp.json.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "name": "t4_msg1",
                            "author": "alice",
                            "dest": "test_user",
                            "body": "hey there, how are you doing today?",
                            "created_utc": 1700000000.0,
                        },
                    },
                ],
            },
        }
        mock_requests.request.return_value = inbox_resp

        convos = provider.list_conversations(limit=10)
        assert len(convos) == 1
        assert isinstance(convos[0], Conversation)
        assert convos[0].id == "t4_msg1"
        assert convos[0].platform == "reddit"
        assert "alice" in convos[0].participants

    def test_get_messages(self, mock_requests):
        provider = _make_provider(mock_requests)

        msg_resp = MagicMock(status_code=200)
        msg_resp.json.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "name": "t4_m1",
                            "author": "alice",
                            "body": "hello!",
                            "created_utc": 1700000000.0,
                        },
                    },
                ],
            },
        }
        mock_requests.request.return_value = msg_resp

        msgs = provider.get_messages("t4_msg1", limit=5)
        assert len(msgs) == 1
        assert isinstance(msgs[0], DirectMessage)
        assert msgs[0].sender == "alice"
        assert msgs[0].content == "hello!"

    def test_list_conversations_empty(self, mock_requests):
        provider = _make_provider(mock_requests)

        empty_resp = MagicMock(status_code=200)
        empty_resp.json.return_value = {"data": {"children": []}}
        mock_requests.request.return_value = empty_resp

        assert provider.list_conversations() == []


# -- TestRateLimitRetry -----------------------------------------------------


@patch("homie_core.social_media.reddit_provider.time")
@patch("homie_core.social_media.reddit_provider.requests")
class TestRateLimitRetry:
    def test_rate_limit_retry(self, mock_requests, mock_time):
        provider = RedditProvider()
        provider._token = "tok123"
        provider._connected = True

        rate_resp = MagicMock(status_code=429)
        rate_resp.headers = {"Retry-After": "1"}

        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = _me_response()

        mock_requests.request.side_effect = [rate_resp, ok_resp]

        result = provider._call("GET", "/api/v1/me")

        assert result["name"] == "test_user"
        assert mock_requests.request.call_count == 2
        mock_time.sleep.assert_called_once_with(1.0)

    def test_rate_limit_exhausted_raises(self, mock_requests, mock_time):
        provider = RedditProvider()
        provider._token = "tok123"
        provider._connected = True

        rate_resp = MagicMock(status_code=429)
        rate_resp.headers = {"Retry-After": "1"}
        rate_resp.raise_for_status.side_effect = _real_requests.HTTPError("429")

        mock_requests.request.return_value = rate_resp

        with pytest.raises(_real_requests.HTTPError):
            provider._call("GET", "/api/v1/me", retries=1)


# -- TestConvenienceMethods -------------------------------------------------


@patch("homie_core.social_media.reddit_provider.requests")
class TestConvenienceMethods:
    """Tests for get_subreddit_feed, get_user_posts, search_subreddit."""

    def test_get_subreddit_feed(self, mock_requests):
        provider = _make_provider(mock_requests)

        feed_resp = MagicMock(status_code=200)
        feed_resp.json.return_value = _listing([SAMPLE_POST, SAMPLE_LINK_POST])
        mock_requests.request.return_value = feed_resp

        posts = provider.get_subreddit_feed("python", count=10)
        assert len(posts) == 2
        assert posts[0]["name"] == "t3_abc123"
        assert posts[1]["name"] == "t3_link1"

        # Verify the correct subreddit URL was called
        call_args = mock_requests.request.call_args
        assert call_args[0] == ("GET", "https://oauth.reddit.com/r/python/hot")

    def test_get_subreddit_feed_empty(self, mock_requests):
        provider = _make_provider(mock_requests)

        resp = MagicMock(status_code=200)
        resp.json.return_value = _listing([])
        mock_requests.request.return_value = resp

        assert provider.get_subreddit_feed("emptysubreddit") == []

    def test_get_subreddit_feed_error_returns_empty(self, mock_requests):
        provider = _make_provider(mock_requests)
        mock_requests.request.side_effect = Exception("API error")

        assert provider.get_subreddit_feed("python") == []

    def test_get_user_posts(self, mock_requests):
        provider = _make_provider(mock_requests)

        posts_resp = MagicMock(status_code=200)
        posts_resp.json.return_value = _listing([SAMPLE_POST])
        mock_requests.request.return_value = posts_resp

        posts = provider.get_user_posts(count=5)
        assert len(posts) == 1
        assert posts[0]["author"] == "poster"

        # Verify the correct user URL was called
        call_args = mock_requests.request.call_args
        assert "/user/test_user/submitted" in call_args[0][1]

    def test_get_user_posts_error_returns_empty(self, mock_requests):
        provider = _make_provider(mock_requests)
        mock_requests.request.side_effect = Exception("fail")

        assert provider.get_user_posts() == []

    def test_search_subreddit(self, mock_requests):
        provider = _make_provider(mock_requests)

        search_resp = MagicMock(status_code=200)
        search_resp.json.return_value = _listing([SAMPLE_POST])
        mock_requests.request.return_value = search_resp

        results = provider.search_subreddit("cool", subreddit="python", count=5)
        assert len(results) == 1
        assert results[0]["name"] == "t3_abc123"

        call_args = mock_requests.request.call_args
        assert "/r/python/search" in call_args[0][1]
        assert call_args[1]["params"]["q"] == "cool"
        assert call_args[1]["params"]["restrict_sr"] == "on"

    def test_search_subreddit_defaults_to_all(self, mock_requests):
        provider = _make_provider(mock_requests)

        search_resp = MagicMock(status_code=200)
        search_resp.json.return_value = _listing([])
        mock_requests.request.return_value = search_resp

        provider.search_subreddit("query")

        call_args = mock_requests.request.call_args
        assert "/r/all/search" in call_args[0][1]

    def test_search_subreddit_error_returns_empty(self, mock_requests):
        provider = _make_provider(mock_requests)
        mock_requests.request.side_effect = Exception("fail")

        assert provider.search_subreddit("q") == []


# -- TestAuthTokenManagement ------------------------------------------------


class TestAuthTokenManagement:
    """Verify token lifecycle and header usage."""

    def test_token_initially_none(self):
        provider = RedditProvider()
        assert provider._token is None
        assert not provider.is_connected

    @patch("homie_core.social_media.reddit_provider.requests")
    def test_bearer_token_in_headers(self, mock_requests):
        provider = _make_provider(mock_requests)

        resp = MagicMock(status_code=200)
        resp.json.return_value = _listing([])
        mock_requests.request.return_value = resp

        provider.get_feed(limit=1)

        call_args = mock_requests.request.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer tok123"
        assert "User-Agent" in headers

    @patch("homie_core.social_media.reddit_provider.requests")
    def test_user_agent_set(self, mock_requests):
        provider = _make_provider(mock_requests)

        resp = MagicMock(status_code=200)
        resp.json.return_value = _listing([])
        mock_requests.request.return_value = resp

        provider.get_feed(limit=1)

        call_args = mock_requests.request.call_args
        assert call_args[1]["headers"]["User-Agent"] == "Homie/1.0"


# -- TestPostFromListing ----------------------------------------------------


class TestPostFromListing:
    def test_self_post(self):
        post = RedditProvider._post_from_listing(SAMPLE_POST)
        assert isinstance(post, SocialPost)
        assert post.id == "t3_abc123"
        assert post.author == "poster"
        assert post.content == "Some body text"
        assert post.likes == 42
        assert post.comments == 7
        assert post.post_type == "text"
        assert post.platform == "reddit"
        assert "reddit.com" in post.url

    def test_link_post_uses_title_as_content(self):
        post = RedditProvider._post_from_listing(SAMPLE_LINK_POST)
        assert post.content == "External link"  # selftext is empty, falls back to title
        assert post.post_type == "link"

    def test_missing_fields_default(self):
        post = RedditProvider._post_from_listing({})
        assert post.id == ""
        assert post.author == ""
        assert post.likes == 0
        assert post.comments == 0
