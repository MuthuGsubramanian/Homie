"""Tests for the Instagram provider."""
from unittest.mock import MagicMock, patch

from homie_core.social_media.instagram_provider import InstagramProvider


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


@patch("homie_core.social_media.instagram_provider.requests")
class TestInstagramConnect:
    def test_connect_success(self, mock_requests):
        mock_requests.request.return_value = _mock_response(
            json_data={"id": "456", "username": "alice_ig"}
        )
        provider = InstagramProvider()
        cred = MagicMock()
        cred.access_token = "tok"
        cred.is_business = True
        assert provider.connect(cred) is True
        assert provider.is_connected is True
        assert provider._is_business is True


@patch("homie_core.social_media.instagram_provider.requests")
class TestInstagramFeed:
    def test_get_feed(self, mock_requests):
        mock_requests.request.return_value = _mock_response(
            json_data={
                "data": [
                    {
                        "id": "media1",
                        "caption": "sunset",
                        "timestamp": "2024-01-01",
                        "media_type": "IMAGE",
                        "permalink": "https://ig/p/1",
                        "like_count": 10,
                        "comments_count": 3,
                    }
                ]
            }
        )
        provider = InstagramProvider()
        provider._token = "tok"
        posts = provider.get_feed(limit=5)
        assert len(posts) == 1
        assert posts[0].id == "media1"
        assert posts[0].content == "sunset"
        assert posts[0].likes == 10
        assert posts[0].post_type == "image"


@patch("homie_core.social_media.instagram_provider.requests")
class TestInstagramProfile:
    def test_get_profile(self, mock_requests):
        mock_requests.request.return_value = _mock_response(
            json_data={
                "id": "456",
                "username": "alice_ig",
                "name": "Alice",
                "biography": "hello world",
                "profile_picture_url": "http://pic",
                "media_count": 42,
            }
        )
        provider = InstagramProvider()
        provider._token = "tok"
        profile = provider.get_profile()
        assert profile.username == "alice_ig"
        assert profile.bio == "hello world"
        assert profile.platform == "instagram"


@patch("homie_core.social_media.instagram_provider.requests")
class TestInstagramPublish:
    def test_publish_requires_media(self, mock_requests):
        provider = InstagramProvider()
        provider._token = "tok"
        provider._is_business = True
        result = provider.publish("Nice day!")
        assert result["status"] == "error"
        assert "media" in result["error"].lower()

    def test_publish_non_business_error(self, mock_requests):
        provider = InstagramProvider()
        provider._token = "tok"
        provider._is_business = False
        result = provider.publish("Nice day!", media_urls=["http://img.png"])
        assert result["status"] == "error"
        assert "business" in result["error"].lower()


@patch("homie_core.social_media.instagram_provider.requests")
class TestInstagramDM:
    def test_dm_business_account_required(self, mock_requests):
        provider = InstagramProvider()
        provider._token = "tok"
        provider._is_business = False

        result = provider.send_message("user1", "hi")
        assert result["status"] == "error"
        assert "business" in result["error"].lower()

        convos = provider.list_conversations()
        assert convos == []

        msgs = provider.get_messages("conv1")
        assert msgs == []
