from __future__ import annotations

import platform
import socket
from datetime import datetime
from typing import Any, Dict

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


def _tailscale_ip() -> str:
    # Best-effort: prefer 100.x address
    for addr in socket.getaddrinfo(socket.gethostname(), None):
        ip = addr[4][0]
        if ip.startswith("100."):
            return ip
    return "unknown"


def collect_status() -> Dict[str, Any]:
    cpu = psutil.cpu_percent(interval=0.1) if psutil else None
    mem = psutil.virtual_memory()._asdict() if psutil else {}
    disk = psutil.disk_usage("/") if psutil else None
    gpu_present = False  # placeholder; extend with nvidia-smi check

    return {
        "ts": datetime.utcnow().isoformat() + "Z",
        "os": platform.platform(),
        "cpu_pct": cpu,
        "ram": {"total": mem.get("total"), "used": mem.get("used"), "percent": mem.get("percent")},
        "disk": {"total": getattr(disk, "total", None), "free": getattr(disk, "free", None)},
        "gpu": gpu_present,
        "docker": _check_docker(),
        "tailscale_ip": _tailscale_ip(),
    }


def _check_docker() -> bool:
    try:
        import docker  # type: ignore

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


__all__ = ["collect_status"]
