from __future__ import annotations

import hashlib
import hmac
import time
from typing import Callable

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class HMACAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware enforcing HMAC headers on all non-health routes."""

    def __init__(self, app, shared_secret: str, skew_seconds: int = 60):
        super().__init__(app)
        self.shared_secret = shared_secret
        self.skew_seconds = skew_seconds

    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path in {"/health"}:
            return await call_next(request)

        client_host = request.client.host if request.client else ""
        if client_host and not (client_host.startswith("100.") or client_host.startswith("127.") or client_host == "::1"):
            raise HTTPException(status_code=403, detail="tailnet/localhost only")

        ts_header = request.headers.get("X-HOMIE-TS")
        sig_header = request.headers.get("X-HOMIE-SIG")
        if not ts_header or not sig_header:
            raise HTTPException(status_code=401, detail="missing auth headers")

        try:
            ts = int(ts_header)
        except ValueError:
            raise HTTPException(status_code=401, detail="bad timestamp")

        if abs(time.time() - ts) > self.skew_seconds:
            raise HTTPException(status_code=401, detail="stale request")

        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8") if body_bytes else "{}"
        msg = f"{request.method}|{request.url.path}|{ts}|{body_str}".encode("utf-8")
        expected = hmac.new(self.shared_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig_header):
            raise HTTPException(status_code=401, detail="invalid signature")

        return await call_next(request)


__all__ = ["HMACAuthMiddleware"]
