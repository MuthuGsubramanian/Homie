from __future__ import annotations

from typing import Dict, List

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore


def active_window_title() -> str | None:
    # Placeholder: platform specific; intentionally not implemented to avoid hidden capture.
    return None


def process_summary(limit: int = 5) -> List[Dict]:
    if not psutil:
        return []
    procs = sorted(psutil.process_iter(["name", "cpu_percent"]), key=lambda p: p.info.get("cpu_percent", 0), reverse=True)
    summary = []
    for proc in procs[:limit]:
        info = proc.info
        summary.append({"name": info.get("name"), "cpu_percent": info.get("cpu_percent")})
    return summary


__all__ = ["active_window_title", "process_summary"]
