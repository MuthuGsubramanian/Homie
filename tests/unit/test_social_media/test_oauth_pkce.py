from unittest.mock import patch, MagicMock
from homie_core.social_media.oauth import SocialMediaOAuth


class TestPKCE:
    def test_pkce_disabled_by_default(self):
        oauth = SocialMediaOAuth(
            platform="test", client_id="id", client_secret="secret",
            auth_url="https://example.com/auth", token_url="https://example.com/token",
            scopes=["read"], redirect_port=8551,
        )
        assert oauth._use_pkce is False
        assert oauth._code_verifier is None

    def test_pkce_enabled_generates_verifier(self):
        oauth = SocialMediaOAuth(
            platform="test", client_id="id", client_secret="secret",
            auth_url="https://example.com/auth", token_url="https://example.com/token",
            scopes=["read"], redirect_port=8551, use_pkce=True,
        )
        assert oauth._use_pkce is True
        assert oauth._code_verifier is not None
        assert 43 <= len(oauth._code_verifier) <= 128

    def test_pkce_auth_url_includes_challenge(self):
        oauth = SocialMediaOAuth(
            platform="test", client_id="id", client_secret="secret",
            auth_url="https://example.com/auth", token_url="https://example.com/token",
            scopes=["read"], redirect_port=8551, use_pkce=True,
        )
        url = oauth.get_auth_url()
        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url

    def test_pkce_exchange_sends_verifier(self):
        oauth = SocialMediaOAuth(
            platform="test", client_id="id", client_secret="secret",
            auth_url="https://example.com/auth", token_url="https://example.com/token",
            scopes=["read"], redirect_port=8551, use_pkce=True,
        )
        with patch("homie_core.social_media.oauth.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"access_token": "tok", "token_type": "bearer"},
            )
            mock_post.return_value.raise_for_status = MagicMock()
            oauth.exchange("test_code")
            call_data = mock_post.call_args.kwargs["data"]
            assert "code_verifier" in call_data

    def test_public_client_omits_secret(self):
        oauth = SocialMediaOAuth(
            platform="test", client_id="id", client_secret="",
            auth_url="https://example.com/auth", token_url="https://example.com/token",
            scopes=["read"], redirect_port=8551, use_pkce=True, is_public_client=True,
        )
        with patch("homie_core.social_media.oauth.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"access_token": "tok", "token_type": "bearer"},
            )
            mock_post.return_value.raise_for_status = MagicMock()
            oauth.exchange("test_code")
            call_data = mock_post.call_args.kwargs["data"]
            assert "client_secret" not in call_data

    def test_extra_params_in_auth_url(self):
        oauth = SocialMediaOAuth(
            platform="test", client_id="id", client_secret="secret",
            auth_url="https://example.com/auth", token_url="https://example.com/token",
            scopes=["read"], redirect_port=8551,
        )
        url = oauth.get_auth_url(extra_params={"duration": "permanent"})
        assert "duration=permanent" in url
