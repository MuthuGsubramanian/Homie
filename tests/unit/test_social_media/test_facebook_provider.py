"""Tests for the Facebook provider."""
from unittest.mock import MagicMock, patch

from homie_core.social_media.facebook_provider import FacebookProvider


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


@patch("homie_core.social_media.facebook_provider.requests")
class TestFacebookConnect:
    def test_connect_success(self, mock_requests):
        mock_requests.request.return_value = _mock_response(
            json_data={"id": "123", "name": "Alice"}
        )
        provider = FacebookProvider()
        cred = MagicMock()
        cred.access_token = "tok"
        assert provider.connect(cred) is True
        assert provider.is_connected is True

    def test_connect_failure(self, mock_requests):
        mock_requests.request.side_effect = Exception("network")
        provider = FacebookProvider()
        cred = MagicMock()
        cred.access_token = "tok"
        assert provider.connect(cred) is False
        assert provider.is_connected is False


@patch("homie_core.social_media.facebook_provider.requests")
class TestFacebookFeed:
    def test_get_feed(self, mock_requests):
        mock_requests.request.return_value = _mock_response(
            json_data={
                "data": [
                    {
                        "id": "post1",
                        "message": "hello",
                        "created_time": "2024-01-01",
                        "from": {"name": "Bob"},
                        "likes": {"summary": {"total_count": 5}},
                        "comments": {"summary": {"total_count": 2}},
                    }
                ]
            }
        )
        provider = FacebookProvider()
        provider._token = "tok"
        posts = provider.get_feed(limit=5)
        assert len(posts) == 1
        assert posts[0].id == "post1"
        assert posts[0].likes == 5
        assert posts[0].comments == 2


@patch("homie_core.social_media.facebook_provider.requests")
class TestFacebookProfile:
    def test_get_profile(self, mock_requests):
        mock_requests.request.return_value = _mock_response(
            json_data={
                "id": "123",
                "name": "Alice",
                "email": "a@b.com",
                "picture": {"data": {"url": "http://pic"}},
            }
        )
        provider = FacebookProvider()
        provider._token = "tok"
        profile = provider.get_profile()
        assert profile.display_name == "Alice"
        assert profile.avatar_url == "http://pic"
        assert profile.platform == "facebook"


@patch("homie_core.social_media.facebook_provider.requests")
class TestFacebookPublish:
    def test_publish(self, mock_requests):
        mock_requests.request.return_value = _mock_response(
            json_data={"id": "new_post_1"}
        )
        provider = FacebookProvider()
        provider._token = "tok"
        result = provider.publish("Hello world!")
        assert result["status"] == "ok"
        assert result["id"] == "new_post_1"


@patch("homie_core.social_media.facebook_provider.requests")
class TestFacebookMessages:
    def test_send_message(self, mock_requests):
        mock_requests.request.return_value = _mock_response(
            json_data={"id": "msg1"}
        )
        provider = FacebookProvider()
        provider._token = "tok"
        result = provider.send_message("conv123", "Hi")
        assert result["status"] == "sent"
        mock_requests.request.assert_called_once()


@patch("homie_core.social_media.facebook_provider.requests")
class TestFacebookRateLimit:
    def test_rate_limit_retry(self, mock_requests):
        rate_limited = _mock_response(status_code=429)
        ok = _mock_response(json_data={"id": "123", "name": "Alice"})
        mock_requests.request.side_effect = [rate_limited, ok]

        provider = FacebookProvider()
        provider._token = "tok"
        resp = provider._call("GET", "/me")
        assert resp.status_code == 200
        assert mock_requests.request.call_count == 2
