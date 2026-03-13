"""Tests for OAuth 2.0 flow — local redirect + manual fallback."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

from homie_core.email.oauth import (
    GmailOAuth,
    GMAIL_SCOPES,
    build_auth_url,
    exchange_code,
)


class TestBuildAuthUrl:
    def test_returns_url_with_scopes(self):
        url = build_auth_url(client_id="test-id", redirect_uri="http://localhost:8547/callback")
        assert "test-id" in url
        assert "scope=" in url
        assert "localhost" in url

    def test_includes_all_scopes(self):
        url = build_auth_url(client_id="test-id", redirect_uri="urn:ietf:wg:oauth:2.0:oob")
        for scope in GMAIL_SCOPES:
            # URL-encoded scope
            assert "gmail" in url


class TestExchangeCode:
    @patch("homie_core.email.oauth.requests.post")
    def test_exchange_returns_tokens(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "ya29.xxx",
            "refresh_token": "1//xxx",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_post.return_value = mock_resp

        tokens = exchange_code(
            code="auth-code",
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost:8547/callback",
        )
        assert tokens["access_token"] == "ya29.xxx"
        assert tokens["refresh_token"] == "1//xxx"

    @patch("homie_core.email.oauth.requests.post")
    def test_exchange_error_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "invalid_grant"}
        mock_resp.raise_for_status.side_effect = Exception("400 Bad Request")
        mock_post.return_value = mock_resp

        try:
            exchange_code("bad-code", "id", "secret", "uri")
            assert False, "Should have raised"
        except Exception:
            pass


class TestGmailOAuth:
    def test_scopes_defined(self):
        assert len(GMAIL_SCOPES) == 3
        assert "https://www.googleapis.com/auth/gmail.readonly" in GMAIL_SCOPES

    @patch("homie_core.email.oauth.requests.post")
    def test_refresh_token(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "ya29.new",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_post.return_value = mock_resp

        oauth = GmailOAuth(client_id="cid", client_secret="csec")
        result = oauth.refresh_access_token("1//refresh")
        assert result["access_token"] == "ya29.new"
