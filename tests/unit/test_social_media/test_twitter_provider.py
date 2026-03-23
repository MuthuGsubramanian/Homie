"""Tests for the Twitter/X social media provider."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests as _real_requests

from homie_core.social_media.models import ProfileInfo, ProfileStats, SocialPost
from homie_core.social_media.twitter_provider import TwitterProvider


# -- helpers ----------------------------------------------------------------


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


SAMPLE_TWEET = {
    "id": "t1",
    "text": "Hello world",
    "author_id": "123",
    "created_at": 1700000000.0,
    "public_metrics": {
        "like_count": 10,
        "reply_count": 2,
        "retweet_count": 5,
    },
}


# -- TestTwitterConnect -----------------------------------------------------


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

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_connect_stores_refresh_token(self, mock_requests: MagicMock) -> None:
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"data": {"id": "1", "username": "u"}}
        mock_requests.get.return_value = resp

        provider = TwitterProvider()
        cred = SimpleNamespace(access_token="at", refresh_token="rt_xyz")
        provider.connect(cred)
        assert provider._refresh_token_str == "rt_xyz"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_connect_no_refresh_token(self, mock_requests: MagicMock) -> None:
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"data": {"id": "1", "username": "u"}}
        mock_requests.get.return_value = resp

        provider = TwitterProvider()
        cred = SimpleNamespace(access_token="at")
        provider.connect(cred)
        assert provider._refresh_token_str is None


# -- TestTwitterFeed --------------------------------------------------------


class TestTwitterFeed:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_feed(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        feed_resp = MagicMock(status_code=200)
        feed_resp.json.return_value = {"data": [SAMPLE_TWEET]}
        mock_requests.get.return_value = feed_resp

        posts = provider.get_feed(limit=5)
        assert len(posts) == 1
        assert isinstance(posts[0], SocialPost)
        assert posts[0].id == "t1"
        assert posts[0].likes == 10
        assert posts[0].platform == "twitter"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_feed_empty(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        feed_resp = MagicMock(status_code=200)
        feed_resp.json.return_value = {"data": []}
        mock_requests.get.return_value = feed_resp

        assert provider.get_feed(limit=5) == []

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_feed_error_returns_empty(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)
        mock_requests.get.side_effect = Exception("API down")

        assert provider.get_feed(limit=5) == []


# -- TestTwitterSearch ------------------------------------------------------


class TestTwitterSearch:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_search_posts(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        search_resp = MagicMock(status_code=200)
        search_resp.json.return_value = {"data": [SAMPLE_TWEET]}
        mock_requests.get.return_value = search_resp

        posts = provider.search_posts("python", limit=10)
        assert len(posts) == 1
        assert posts[0].content == "Hello world"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_search_posts_error_returns_empty(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)
        mock_requests.get.side_effect = Exception("timeout")
        assert provider.search_posts("q") == []


# -- TestTwitterProfile -----------------------------------------------------


class TestTwitterProfile:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_profile_by_username(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        profile_resp = MagicMock(status_code=200)
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
        assert info.profile_url == "https://x.com/alice"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_profile_self(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        me_resp = MagicMock(status_code=200)
        me_resp.json.return_value = {
            "data": {
                "id": "123",
                "username": "testuser",
                "name": "Test",
                "description": "hi",
            },
        }
        mock_requests.get.return_value = me_resp

        info = provider.get_profile()  # None means self
        assert info.username == "testuser"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_profile_error_returns_empty(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)
        mock_requests.get.side_effect = Exception("fail")

        info = provider.get_profile()
        assert info.username == ""
        assert info.platform == "twitter"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_stats(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        stats_resp = MagicMock(status_code=200)
        stats_resp.json.return_value = {
            "data": {
                "public_metrics": {
                    "followers_count": 500,
                    "following_count": 100,
                    "tweet_count": 1200,
                    "listed_count": 10,
                },
            },
        }
        mock_requests.get.return_value = stats_resp

        stats = provider.get_stats()
        assert isinstance(stats, ProfileStats)
        assert stats.followers == 500
        assert stats.following == 100
        assert stats.post_count == 1200
        assert stats.engagement_rate == pytest.approx(10 / 500)

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_stats_error_returns_defaults(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)
        mock_requests.get.side_effect = Exception("fail")

        stats = provider.get_stats()
        assert stats.followers == 0
        assert stats.platform == "twitter"


# -- TestTwitterPublish -----------------------------------------------------


class TestTwitterPublish:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_publish(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        pub_resp = MagicMock(status_code=201)
        pub_resp.json.return_value = {"data": {"id": "new_tweet_1"}}
        mock_requests.post.return_value = pub_resp

        result = provider.publish("Check this out!")
        assert result["status"] == "published"
        assert result["post_id"] == "new_tweet_1"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_publish_error(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)
        mock_requests.post.side_effect = Exception("forbidden")

        result = provider.publish("will fail")
        assert result["status"] == "error"
        assert result["platform"] == "twitter"


# -- TestTwitterDM ----------------------------------------------------------


class TestTwitterDM:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_send_message(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        dm_resp = MagicMock(status_code=200)
        dm_resp.json.return_value = {"data": {"id": "dm1"}}
        mock_requests.post.return_value = dm_resp

        result = provider.send_message("bob", "hey!")
        assert result["status"] == "sent"
        assert result["recipient"] == "bob"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_send_message_error(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)
        mock_requests.post.side_effect = Exception("network")

        result = provider.send_message("bob", "hey!")
        assert result["status"] == "error"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_list_conversations(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        conv_resp = MagicMock(status_code=200)
        conv_resp.json.return_value = {
            "data": [
                {
                    "id": "c1",
                    "participants": ["alice", "bob"],
                    "last_message": "see you",
                    "last_activity": 1700000000.0,
                },
            ],
        }
        mock_requests.get.return_value = conv_resp

        convos = provider.list_conversations()
        assert len(convos) == 1
        assert convos[0].id == "c1"
        assert convos[0].platform == "twitter"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_messages(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        msg_resp = MagicMock(status_code=200)
        msg_resp.json.return_value = {
            "data": [
                {
                    "id": "m1",
                    "sender_id": "alice",
                    "text": "hello",
                    "created_at": 1700000000.0,
                },
            ],
        }
        mock_requests.get.return_value = msg_resp

        msgs = provider.get_messages("c1")
        assert len(msgs) == 1
        assert msgs[0].sender == "alice"
        assert msgs[0].content == "hello"


# -- TestTwitterRateLimit ---------------------------------------------------


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

    @patch("homie_core.social_media.twitter_provider.time.sleep")
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_rate_limit_default_wait(
        self, mock_requests: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        """When Retry-After header is missing, fall back to 5 seconds."""
        rate_resp = MagicMock(status_code=429, headers={})
        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {"data": {}}

        mock_requests.get.side_effect = [rate_resp, ok_resp]

        provider = TwitterProvider()
        provider._token = "tok"
        provider._call("GET", "/users/me")

        mock_sleep.assert_called_once_with(5)

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_http_error_raises(self, mock_requests: MagicMock) -> None:
        err_resp = MagicMock(status_code=403)
        err_resp.raise_for_status.side_effect = _real_requests.HTTPError("Forbidden")
        mock_requests.get.return_value = err_resp

        provider = TwitterProvider()
        provider._token = "tok"
        with pytest.raises(_real_requests.HTTPError):
            provider._call("GET", "/users/me")


# -- TestTwitterConvenienceMethods -----------------------------------------


class TestTwitterConvenienceMethods:
    """Tests for get_timeline, get_mentions, post_tweet, search."""

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_timeline(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        timeline_resp = MagicMock(status_code=200)
        timeline_resp.json.return_value = {"data": [SAMPLE_TWEET]}
        mock_requests.get.return_value = timeline_resp

        tweets = provider.get_timeline(count=5)
        assert len(tweets) == 1
        assert tweets[0]["id"] == "t1"
        assert tweets[0]["text"] == "Hello world"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_timeline_empty(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        resp = MagicMock(status_code=200)
        resp.json.return_value = {"data": []}
        mock_requests.get.return_value = resp

        assert provider.get_timeline(count=10) == []

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_timeline_error_returns_empty(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)
        mock_requests.get.side_effect = Exception("fail")
        assert provider.get_timeline() == []

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_mentions(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        mentions_resp = MagicMock(status_code=200)
        mentions_resp.json.return_value = {
            "data": [
                {"id": "m1", "text": "@testuser hi!", "author_id": "999"},
            ],
        }
        mock_requests.get.return_value = mentions_resp

        mentions = provider.get_mentions(count=10)
        assert len(mentions) == 1
        assert mentions[0]["text"] == "@testuser hi!"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_mentions_error_returns_empty(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)
        mock_requests.get.side_effect = Exception("fail")
        assert provider.get_mentions() == []

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_post_tweet(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        tweet_resp = MagicMock(status_code=201)
        tweet_resp.json.return_value = {
            "data": {"id": "new1", "text": "My tweet"},
        }
        mock_requests.post.return_value = tweet_resp

        result = provider.post_tweet("My tweet")
        assert result["id"] == "new1"
        assert result["text"] == "My tweet"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_post_tweet_error_returns_empty(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)
        mock_requests.post.side_effect = Exception("fail")
        assert provider.post_tweet("fail") == {}

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_search(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        search_resp = MagicMock(status_code=200)
        search_resp.json.return_value = {"data": [SAMPLE_TWEET]}
        mock_requests.get.return_value = search_resp

        results = provider.search("python", count=10)
        assert len(results) == 1
        assert results[0]["id"] == "t1"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_search_error_returns_empty(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)
        mock_requests.get.side_effect = Exception("fail")
        assert provider.search("q") == []


# -- TestTwitterAuthTokenManagement -----------------------------------------


class TestTwitterAuthTokenManagement:
    """Verify that the bearer token is passed correctly in API calls."""

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_bearer_token_in_headers(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        resp = MagicMock(status_code=200)
        resp.json.return_value = {"data": []}
        mock_requests.get.return_value = resp

        provider.get_timeline(count=5)

        # Check the Authorization header was set with Bearer token
        call_args = mock_requests.get.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers["Authorization"] == "Bearer tok123"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_post_uses_bearer_token(self, mock_requests: MagicMock) -> None:
        provider = _make_provider(mock_requests)

        resp = MagicMock(status_code=201)
        resp.json.return_value = {"data": {"id": "x"}}
        mock_requests.post.return_value = resp

        provider.post_tweet("test")

        call_args = mock_requests.post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers["Authorization"] == "Bearer tok123"

    def test_token_initially_none(self) -> None:
        provider = TwitterProvider()
        assert provider._token is None
        assert not provider.is_connected

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_token_stored_after_connect(self, mock_requests: MagicMock) -> None:
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"data": {"id": "1", "username": "u"}}
        mock_requests.get.return_value = resp

        provider = TwitterProvider()
        provider.connect(SimpleNamespace(access_token="secret_token"))
        assert provider._token == "secret_token"


# -- TestTweetToPost --------------------------------------------------------


class TestTweetToPost:
    def test_tweet_to_post_full(self) -> None:
        post = TwitterProvider._tweet_to_post(SAMPLE_TWEET)
        assert post.id == "t1"
        assert post.platform == "twitter"
        assert post.author == "123"
        assert post.content == "Hello world"
        assert post.likes == 10
        assert post.comments == 2
        assert post.shares == 5
        assert post.url == "https://x.com/i/status/t1"

    def test_tweet_to_post_missing_metrics(self) -> None:
        tweet = {"id": "t2", "text": "bare tweet"}
        post = TwitterProvider._tweet_to_post(tweet)
        assert post.likes == 0
        assert post.comments == 0
        assert post.shares == 0
