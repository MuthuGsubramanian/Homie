"""Structured logging setup for Homie.

Provides file rotation, optional JSON output, and per-module level overrides
so noisy libraries can be silenced without losing visibility into Homie internals.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CONSOLE_FMT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_FILE_FMT = "%(asctime)s [%(levelname)-8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"

# Libraries that are typically noisy at DEBUG/INFO level.
_NOISY_LOGGERS = (
    "urllib3",
    "httpcore",
    "httpx",
    "matplotlib",
    "PIL",
    "asyncio",
)


def setup_logging(
    log_dir: Path,
    level: str = "INFO",
    structured: bool = True,
) -> None:
    """Configure logging with file rotation and optional structured JSON output.

    Parameters
    ----------
    log_dir:
        Directory where ``homie.log`` (and rotated backups) will be written.
        Created automatically if it does not exist.
    level:
        Root log level (e.g. ``"DEBUG"``, ``"INFO"``).
    structured:
        If *True* the file handler emits JSON lines; otherwise plain text.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # --- console handler (always human-readable) ---
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT))
    root.addHandler(console)

    # --- rotating file handler ---
    log_file = log_dir / "homie.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    if structured:
        file_handler.setFormatter(_JsonFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(_FILE_FMT))
    root.addHandler(file_handler)

    # --- silence noisy third-party loggers ---
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``homie.``."""
    if not name.startswith("homie."):
        name = f"homie.{name}"
    return logging.getLogger(name)
