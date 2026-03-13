"""Shared OAuth helpers for social media platforms."""
from __future__ import annotations
import base64
import hashlib
import logging
import secrets
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import requests

logger = logging.getLogger(__name__)


class _CallbackHandler(BaseHTTPRequestHandler):
    code: str | None = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        _CallbackHandler.code = params.get("code", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>Success! You can close this tab.</h1>")

    def log_message(self, format, *args):
        pass


class SocialMediaOAuth:
    def __init__(self, platform: str, client_id: str, client_secret: str,
                 auth_url: str, token_url: str, scopes: list[str],
                 redirect_port: int, use_pkce: bool = False,
                 is_public_client: bool = False):
        self.platform = platform
        self._client_id = client_id
        self._client_secret = client_secret
        self._auth_url = auth_url
        self._token_url = token_url
        self._scopes = scopes
        self._redirect_port = redirect_port
        self._redirect_uri = f"http://localhost:{redirect_port}/callback"
        self._use_pkce = use_pkce
        self._is_public_client = is_public_client

        if use_pkce:
            self._code_verifier: str | None = (
                base64.urlsafe_b64encode(secrets.token_bytes(32))
                .rstrip(b"=")
                .decode()
            )
        else:
            self._code_verifier = None

    def _pkce_challenge(self) -> str:
        """Compute S256 code_challenge from the stored code_verifier."""
        digest = hashlib.sha256(self._code_verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    def get_auth_url(self, extra_params: dict[str, str] | None = None) -> str:
        params: dict[str, str] = {
            "client_id": self._client_id,
            "response_type": "code",
            "redirect_uri": self._redirect_uri,
            "scope": " ".join(self._scopes),
        }
        if self._use_pkce:
            params["code_challenge"] = self._pkce_challenge()
            params["code_challenge_method"] = "S256"
        if extra_params:
            params.update(extra_params)
        return f"{self._auth_url}?{urllib.parse.urlencode(params)}"

    def wait_for_redirect(self, timeout: int = 120) -> str | None:
        _CallbackHandler.code = None
        server = HTTPServer(("127.0.0.1", self._redirect_port), _CallbackHandler)
        server.timeout = timeout
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        server.server_close()
        return _CallbackHandler.code

    def exchange(self, code: str) -> dict[str, Any]:
        data: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._redirect_uri,
            "client_id": self._client_id,
        }
        if not self._is_public_client:
            data["client_secret"] = self._client_secret
        if self._use_pkce and self._code_verifier is not None:
            data["code_verifier"] = self._code_verifier
        resp = requests.post(self._token_url, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        resp = requests.post(self._token_url, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()
