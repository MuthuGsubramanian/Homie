"""System health verification for Homie.

Run pre-flight checks before the assistant starts to surface configuration
and environment problems early with clear, actionable messages.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any


def _result(name: str, passed: bool, message: str = "") -> dict[str, Any]:
    return {"check": name, "passed": passed, "message": message}


class SystemHealthCheck:
    """Pre-flight checks before Homie starts."""

    # Minimum supported Python version.
    MIN_PYTHON = (3, 10)

    # Optional dependencies that enable extra features.
    OPTIONAL_DEPS = (
        "rich",
        "pyttsx3",
        "whisper",
        "torch",
        "fastapi",
    )

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_python_version(self) -> dict[str, Any]:
        """Ensure the running Python meets the minimum version."""
        current = sys.version_info[:2]
        ok = current >= self.MIN_PYTHON
        msg = (
            f"Python {current[0]}.{current[1]}"
            if ok
            else (
                f"Python {current[0]}.{current[1]} is below minimum "
                f"{self.MIN_PYTHON[0]}.{self.MIN_PYTHON[1]}"
            )
        )
        return _result("python_version", ok, msg)

    def check_disk_space(self, min_gb: float = 1.0) -> dict[str, Any]:
        """Check that at least *min_gb* GB of free disk space is available."""
        path = self.config.get("data_dir", tempfile.gettempdir())
        try:
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024 ** 3)
            ok = free_gb >= min_gb
            msg = f"{free_gb:.1f} GB free" if ok else f"Only {free_gb:.1f} GB free (need {min_gb} GB)"
            return _result("disk_space", ok, msg)
        except OSError as exc:
            return _result("disk_space", False, str(exc))

    def check_model_file(self, path: str) -> dict[str, Any]:
        """Verify that the model file exists and is non-empty."""
        p = Path(path)
        if not p.exists():
            return _result("model_file", False, f"Model file not found: {path}")
        if p.stat().st_size == 0:
            return _result("model_file", False, f"Model file is empty: {path}")
        return _result("model_file", True, f"Model file OK ({p.stat().st_size} bytes)")

    def check_database(self, path: str) -> dict[str, Any]:
        """Ensure we can open (or create) the SQLite database and write to it."""
        try:
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS _health_check (id INTEGER PRIMARY KEY)")
            cur.execute("INSERT INTO _health_check (id) VALUES (1)")
            conn.commit()
            cur.execute("DELETE FROM _health_check WHERE id = 1")
            conn.commit()
            conn.close()
            return _result("database", True, f"SQLite writable at {path}")
        except Exception as exc:
            return _result("database", False, f"SQLite error: {exc}")

    def check_config_valid(self) -> dict[str, Any]:
        """Basic sanity check on the loaded configuration dict."""
        if not self.config:
            return _result("config_valid", False, "Configuration is empty")
        return _result("config_valid", True, "Configuration loaded")

    def check_optional_deps(self) -> dict[str, Any]:
        """Report which optional dependencies are importable."""
        available: list[str] = []
        missing: list[str] = []
        for dep in self.OPTIONAL_DEPS:
            try:
                importlib.import_module(dep)
                available.append(dep)
            except ImportError:
                missing.append(dep)
        msg_parts = []
        if available:
            msg_parts.append(f"available: {', '.join(available)}")
        if missing:
            msg_parts.append(f"missing: {', '.join(missing)}")
        return _result("optional_deps", True, "; ".join(msg_parts))

    # ------------------------------------------------------------------
    # Aggregate runner
    # ------------------------------------------------------------------

    def run_all(self) -> list[dict[str, Any]]:
        """Execute every health check and return the collected results."""
        results: list[dict[str, Any]] = []

        results.append(self.check_python_version())
        results.append(self.check_disk_space())
        results.append(self.check_config_valid())
        results.append(self.check_optional_deps())

        model_path = self.config.get("model_path")
        if model_path:
            results.append(self.check_model_file(model_path))

        db_path = self.config.get("db_path")
        if db_path:
            results.append(self.check_database(db_path))

        return results
