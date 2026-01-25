from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
import ipaddress


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def host_header(name: str) -> str:
    return f"==== {name} ===="


def safe_get_str(data: dict, key: str, default: str = "") -> str:
    val = data.get(key, default)
    return str(val) if val is not None else default


def ensure_ip_literal(host: str) -> str:
    """
    Validate that the provided host string is an IPv4 or IPv6 literal.
    Raises ValueError if not. Returns the normalized string representation.
    """
    try:
        ip_obj = ipaddress.ip_address(host.strip())
        return ip_obj.compressed
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Target host must be an IP address, got '{host}'") from exc


__all__ = [
    "setup_logging",
    "pretty_json",
    "timestamp",
    "host_header",
    "safe_get_str",
    "ensure_ip_literal",
]
