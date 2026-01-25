from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

import requests
from urllib.parse import urlparse
import ipaddress

DEFAULT_TIMEOUT = 10


class NodeApiError(Exception):
    """Raised when node API call fails."""


def _build_signature(secret: str, method: str, path: str, body: str, ts: int) -> str:
    msg = f"{method.upper()}|{path}|{ts}|{body}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


class NodeApiClient:
    """Signed HTTP client for talking to a HOMIE node over Tailnet."""

    def __init__(self, base_url: str, shared_secret: str, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.shared_secret = shared_secret
        self.timeout = timeout
        self._assert_ip_base()

    def _assert_ip_base(self) -> None:
        parsed = urlparse(self.base_url if "://" in self.base_url else f"https://{self.base_url}")
        host = parsed.hostname or ""
        try:
            ipaddress.ip_address(host)
        except Exception as exc:  # noqa: BLE001
            raise NodeApiError(f"Node base_url must use IP, got '{self.base_url}'") from exc

    def _headers(self, method: str, path: str, body: Optional[Dict[str, Any]]) -> Dict[str, str]:
        ts = int(time.time())
        body_str = json.dumps(body or {}, separators=(",", ":"), ensure_ascii=True)
        sig = _build_signature(self.shared_secret, method, path, body_str, ts)
        return {
            "X-HOMIE-TS": str(ts),
            "X-HOMIE-SIG": sig,
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = self._headers(method, path, body)
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=body,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:  # network, DNS, TLS, etc.
            raise NodeApiError(str(exc)) from exc

    def get_status(self) -> Dict[str, Any]:
        # Kept for backwards compatibility with existing nodes
        try:
            return self._request("GET", "/v1/metrics")
        except NodeApiError:
            return self._request("GET", "/status")

    def run_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/v1/run", payload)

    def run_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_task(payload)

    def start_workflow(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/workflows/start", payload)

    def stop_workflow(self, workflow_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/workflows/{workflow_id}/stop")

    def replay_workflow(self, workflow_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", f"/workflows/{workflow_id}/replay", payload)

    def clear(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/clear", payload)

    # v1 endpoints
    def hello(self, controller_instance: str) -> Dict[str, Any]:
        return self._request("POST", "/v1/hello", {"controller": controller_instance})

    def metrics(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/metrics")

    def fetch_logs(self, paths: list[str], tail_lines: int | None = None) -> Dict[str, Any]:
        return self._request("POST", "/v1/fetch_logs", {"paths": paths, "tail_lines": tail_lines})

    def rollback(self, run_id: str) -> Dict[str, Any]:
        return self._request("POST", "/v1/rollback", {"run_id": run_id})

    def upload_recording(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/v1/recordings/upload", {"name": name, "payload": payload})


__all__ = ["NodeApiClient", "NodeApiError"]
