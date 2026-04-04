"""LAN peer inference client — discovers and routes to nearby Homie instances."""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT = 5
_GENERATE_TIMEOUT = 120
_DISCOVERY_INTERVAL = 30


class PeerInfo:
    """Tracks a discovered LAN peer and its health."""

    __slots__ = ("host", "port", "device_id", "device_name", "latency", "last_seen")

    def __init__(
        self,
        host: str,
        port: int,
        device_id: str = "",
        device_name: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.device_id = device_id
        self.device_name = device_name
        self.latency: float = float("inf")
        self.last_seen: float = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __repr__(self) -> str:
        return f"PeerInfo({self.device_id!r}, {self.host}:{self.port}, latency={self.latency:.0f}ms)"


class LANInferenceClient:
    """Discovers Homie peers on the LAN via mDNS and routes inference to the
    best available peer (lowest latency).

    Each peer is expected to expose an OpenAI-compatible ``/v1/chat/completions``
    endpoint and a ``/health`` endpoint for liveness checks.

    Parameters
    ----------
    device_id:
        This instance's unique device identifier (used to avoid
        self-discovery).
    device_name:
        Human-readable name for this device.
    port:
        The port this instance listens on (passed to mDNS advertisement).
    """

    def __init__(
        self,
        device_id: str,
        device_name: str = "",
        port: int = 8765,
    ) -> None:
        self._device_id = device_id
        self._device_name = device_name
        self._port = port
        self._peers: dict[str, PeerInfo] = {}
        self._lock = threading.Lock()
        self._discovery = None
        self._refresh_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin mDNS browsing and periodic health-check refresh."""
        if self._running:
            return
        try:
            from homie_core.network.discovery import HomieDiscovery
        except ImportError:
            logger.warning("Network module unavailable — LAN inference disabled")
            return

        self._discovery = HomieDiscovery(
            device_id=self._device_id,
            device_name=self._device_name,
            port=self._port,
        )
        self._discovery.start_browsing()
        self._running = True
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True, name="lan-peer-refresh"
        )
        self._refresh_thread.start()
        logger.info("LAN inference client started (device=%s)", self._device_id)

    def stop(self) -> None:
        """Stop discovery and background refresh."""
        self._running = False
        if self._discovery:
            self._discovery.stop_browsing()
            self._discovery = None
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
            self._refresh_thread = None
        with self._lock:
            self._peers.clear()
        logger.info("LAN inference client stopped")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True when at least one healthy peer is reachable."""
        with self._lock:
            return any(
                p.latency < float("inf") for p in self._peers.values()
            )

    @property
    def peers(self) -> list[PeerInfo]:
        """Return a snapshot of known peers sorted by latency (best first)."""
        with self._lock:
            return sorted(self._peers.values(), key=lambda p: p.latency)

    # ------------------------------------------------------------------
    # Background refresh
    # ------------------------------------------------------------------

    def _refresh_loop(self) -> None:
        """Periodically refresh the peer list from mDNS and run health checks."""
        while self._running:
            try:
                self._sync_peers()
                self._health_check_all()
            except Exception:
                logger.debug("Peer refresh error", exc_info=True)
            # Sleep in small increments so stop() is responsive
            for _ in range(int(_DISCOVERY_INTERVAL)):
                if not self._running:
                    return
                time.sleep(1)

    def _sync_peers(self) -> None:
        """Pull newly-discovered devices from HomieDiscovery into our peer map."""
        if not self._discovery:
            return
        discovered = self._discovery.discovered_devices
        with self._lock:
            # Add new peers
            for device_id, info in discovered.items():
                if device_id == self._device_id:
                    continue
                if device_id not in self._peers:
                    self._peers[device_id] = PeerInfo(
                        host=info["host"],
                        port=info["port"],
                        device_id=device_id,
                        device_name=info.get("name", ""),
                    )
                    logger.info("Discovered LAN peer: %s at %s:%s",
                                device_id, info["host"], info["port"])
                else:
                    # Update host/port in case they changed
                    peer = self._peers[device_id]
                    peer.host = info["host"]
                    peer.port = info["port"]

            # Remove peers that disappeared from mDNS
            stale = [did for did in self._peers if did not in discovered]
            for did in stale:
                logger.info("LAN peer removed: %s", did)
                del self._peers[did]

    def _health_check_all(self) -> None:
        """Run a health check against every known peer, updating latency."""
        with self._lock:
            peers = list(self._peers.values())
        for peer in peers:
            self._health_check(peer)

    @staticmethod
    def _health_check(peer: PeerInfo) -> None:
        """GET /health on the peer; update latency or mark as unreachable."""
        url = f"{peer.base_url}/health"
        try:
            start = time.monotonic()
            req = Request(url, method="GET")
            with urlopen(req, timeout=_HEALTH_TIMEOUT):
                pass
            elapsed_ms = (time.monotonic() - start) * 1000
            peer.latency = elapsed_ms
            peer.last_seen = time.time()
        except (URLError, OSError, HTTPError):
            peer.latency = float("inf")

    # ------------------------------------------------------------------
    # Peer selection
    # ------------------------------------------------------------------

    def _best_peer(self) -> Optional[PeerInfo]:
        """Return the healthy peer with the lowest latency, or None."""
        with self._lock:
            candidates = [p for p in self._peers.values() if p.latency < float("inf")]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p.latency)

    # ------------------------------------------------------------------
    # Inference — generate (matches QubridClient interface)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Send inference request to the best available LAN peer.

        Raises
        ------
        ConnectionError
            If no peers are available or the request fails.
        RuntimeError
            If the peer returns an API error.
        """
        peer = self._best_peer()
        if peer is None:
            raise ConnectionError("No LAN peers available for inference")

        payload: dict = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        url = f"{peer.base_url}/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=_GENERATE_TIMEOUT) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"].get("content", "") or ""
        except HTTPError as exc:
            # Mark peer as unhealthy and try to re-raise with detail
            peer.latency = float("inf")
            try:
                body = json.loads(exc.read())
                msg = body.get("error", {}).get("message", str(exc))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"LAN peer {peer.device_id} error ({exc.code}): {msg}") from exc
        except (URLError, OSError) as exc:
            peer.latency = float("inf")
            raise ConnectionError(
                f"LAN peer {peer.device_id} unreachable: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Inference — stream (matches QubridClient interface)
    # ------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Stream inference from the best available LAN peer.

        Yields content chunks. Raises ConnectionError when no peers are available.
        """
        peer = self._best_peer()
        if peer is None:
            raise ConnectionError("No LAN peers available for inference")

        payload: dict = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        url = f"{peer.base_url}/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=300) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except HTTPError as exc:
            peer.latency = float("inf")
            try:
                body = json.loads(exc.read())
                msg = body.get("error", {}).get("message", str(exc))
            except Exception:
                msg = str(exc)
            raise RuntimeError(f"LAN peer {peer.device_id} error ({exc.code}): {msg}") from exc
        except (URLError, OSError) as exc:
            peer.latency = float("inf")
            raise ConnectionError(
                f"LAN peer {peer.device_id} unreachable: {exc}"
            ) from exc
