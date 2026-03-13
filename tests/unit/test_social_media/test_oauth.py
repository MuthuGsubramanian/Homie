"""Tests for social media OAuth helpers."""
from unittest.mock import patch, MagicMock
from homie_core.social_media.oauth import SocialMediaOAuth


def _make_oauth(**overrides):
    defaults = dict(
        platform="twitter", client_id="cid", client_secret="csec",
        auth_url="https://example.com/authorize",
        token_url="https://example.com/token",
        scopes=["read", "write"], redirect_port=8551,
    )
    defaults.update(overrides)
    return SocialMediaOAuth(**defaults)


class TestGetAuthUrl:
    def test_contains_client_id(self):
        url = _make_oauth().get_auth_url()
        assert "client_id=cid" in url

    def test_contains_scopes(self):
        url = _make_oauth().get_auth_url()
        assert "read" in url and "write" in url

    def test_contains_redirect_port(self):
        url = _make_oauth(redirect_port=8552).get_auth_url()
        assert "8552" in url

    def test_contains_response_type_code(self):
        url = _make_oauth().get_auth_url()
        assert "response_type=code" in url


class TestExchange:
    @patch("homie_core.social_media.oauth.requests")
    def test_exchange_success(self, mock_req):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"access_token": "tok", "refresh_token": "ref"}
        mock_req.post.return_value = resp

        tokens = _make_oauth().exchange("code123")
        assert tokens["access_token"] == "tok"
        assert tokens["refresh_token"] == "ref"
        mock_req.post.assert_called_once()

    @patch("homie_core.social_media.oauth.requests")
    def test_exchange_sends_correct_data(self, mock_req):
        resp = MagicMock()
        resp.json.return_value = {}
        mock_req.post.return_value = resp

        _make_oauth().exchange("mycode")
        call_kwargs = mock_req.post.call_args
        data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert data["grant_type"] == "authorization_code"
        assert data["code"] == "mycode"
        assert data["client_id"] == "cid"


class TestRefresh:
    @patch("homie_core.social_media.oauth.requests")
    def test_refresh_success(self, mock_req):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"access_token": "new_tok", "refresh_token": "new_ref"}
        mock_req.post.return_value = resp

        tokens = _make_oauth().refresh("old_ref")
        assert tokens["access_token"] == "new_tok"

    @patch("homie_core.social_media.oauth.requests")
    def test_refresh_sends_correct_grant_type(self, mock_req):
        resp = MagicMock()
        resp.json.return_value = {}
        mock_req.post.return_value = resp

        _make_oauth().refresh("ref_tok")
        call_kwargs = mock_req.post.call_args
        data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "ref_tok"
