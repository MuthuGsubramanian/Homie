"""Tests for the LinkedIn social media provider."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from homie_core.social_media.linkedin_provider import LinkedInProvider
from homie_core.social_media.provider import DirectMessageProvider


MOCK_ME_RESPONSE = {
    "id": "abc123",
    "firstName": {"localized": {"en_US": "Jane"}},
    "lastName": {"localized": {"en_US": "Doe"}},
    "headline": {"localized": {"en_US": "Software Engineer"}},
    "vanityName": "janedoe",
    "numConnections": 500,
    "profilePicture": {
        "displayImage~": {
            "elements": [
                {"identifiers": [{"identifier": "https://img.example.com/photo.jpg"}]}
            ]
        }
    },
}


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


# ------------------------------------------------------------------
# connect
# ------------------------------------------------------------------


@patch("homie_core.social_media.linkedin_provider.requests")
def test_connect_success(mock_requests):
    mock_requests.request.return_value = _mock_response(MOCK_ME_RESPONSE)

    provider = LinkedInProvider()
    cred = SimpleNamespace(access_token="tok_123")
    result = provider.connect(cred)

    assert result is True
    assert provider.is_connected
    assert provider._person_id == "abc123"


@patch("homie_core.social_media.linkedin_provider.requests")
def test_connect_failure(mock_requests):
    mock_requests.request.side_effect = Exception("network error")

    provider = LinkedInProvider()
    cred = SimpleNamespace(access_token="bad_token")
    result = provider.connect(cred)

    assert result is False
    assert not provider.is_connected


# ------------------------------------------------------------------
# get_feed
# ------------------------------------------------------------------


@patch("homie_core.social_media.linkedin_provider.requests")
def test_get_feed(mock_requests):
    feed_data = {
        "elements": [
            {
                "id": "post1",
                "author": "urn:li:person:abc",
                "text": "Hello LinkedIn!",
                "created": {"time": 1700000000.0},
                "likes": 10,
                "comments": 2,
                "shares": 1,
            }
        ]
    }
    mock_requests.request.return_value = _mock_response(feed_data)

    provider = LinkedInProvider()
    provider._token = "tok"
    provider._connected = True

    posts = provider.get_feed(limit=5)

    assert len(posts) == 1
    assert posts[0].id == "post1"
    assert posts[0].platform == "linkedin"
    assert posts[0].content == "Hello LinkedIn!"
    assert posts[0].likes == 10


# ------------------------------------------------------------------
# get_profile
# ------------------------------------------------------------------


@patch("homie_core.social_media.linkedin_provider.requests")
def test_get_profile(mock_requests):
    mock_requests.request.return_value = _mock_response(MOCK_ME_RESPONSE)

    provider = LinkedInProvider()
    provider._token = "tok"
    provider._connected = True

    profile = provider.get_profile()

    assert profile.platform == "linkedin"
    assert profile.display_name == "Jane Doe"
    assert profile.bio == "Software Engineer"
    assert profile.username == "janedoe"
    assert profile.avatar_url == "https://img.example.com/photo.jpg"
    assert profile.profile_url == "https://www.linkedin.com/in/janedoe"


# ------------------------------------------------------------------
# get_stats
# ------------------------------------------------------------------


@patch("homie_core.social_media.linkedin_provider.requests")
def test_get_stats(mock_requests):
    mock_requests.request.return_value = _mock_response({"numConnections": 500})

    provider = LinkedInProvider()
    provider._token = "tok"
    provider._connected = True

    stats = provider.get_stats()
    assert stats.platform == "linkedin"
    assert stats.followers == 500


# ------------------------------------------------------------------
# publish
# ------------------------------------------------------------------


@patch("homie_core.social_media.linkedin_provider.requests")
def test_publish(mock_requests):
    mock_requests.request.return_value = _mock_response({"id": "ugc_post_1"})

    provider = LinkedInProvider()
    provider._token = "tok"
    provider._connected = True
    provider._person_id = "abc123"

    result = provider.publish("My first post!")

    assert result == {"id": "ugc_post_1", "status": "published"}

    call_args = mock_requests.request.call_args
    assert call_args[0][0] == "POST"
    body = call_args[1]["json"]
    assert body["author"] == "urn:li:person:abc123"
    share = body["specificContent"]["com.linkedin.ugc.ShareContent"]
    assert share["shareCommentary"]["text"] == "My first post!"
    assert share["shareMediaCategory"] == "NONE"


# ------------------------------------------------------------------
# DM not supported
# ------------------------------------------------------------------


def test_is_not_dm_provider():
    provider = LinkedInProvider()
    assert not isinstance(provider, DirectMessageProvider)
