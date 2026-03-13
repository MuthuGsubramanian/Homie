"""Shared OAuth helpers for social media platforms."""
from __future__ import annotations
import logging
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import requests as _requests

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
                 redirect_port: int):
        self.platform = platform
        self._client_id = client_id
        self._client_secret = client_secret
        self._auth_url = auth_url
        self._token_url = token_url
        self._scopes = scopes
        self._redirect_port = redirect_port
        self._redirect_uri = f"http://localhost:{redirect_port}/callback"

    def get_auth_url(self) -> str:
        params = {
            "client_id": self._client_id,
            "response_type": "code",
            "redirect_uri": self._redirect_uri,
            "scope": " ".join(self._scopes),
        }
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
        resp = _requests.post(self._token_url, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._redirect_uri,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        resp = _requests.post(self._token_url, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()
