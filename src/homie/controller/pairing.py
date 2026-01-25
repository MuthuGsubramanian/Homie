from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Tuple

SECRET_FILE = Path.home() / ".homie" / "shared.secret"


def generate_secret() -> str:
    raw = os.urandom(32)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def write_secret(secret: str, path: Path = SECRET_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secret, encoding="utf-8")
    return path


def load_secret(path: Path = SECRET_FILE) -> str:
    return path.read_text(encoding="utf-8").strip()


def secret_fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def create_and_save() -> Tuple[str, Path]:
    secret = generate_secret()
    path = write_secret(secret)
    return secret, path


__all__ = ["generate_secret", "write_secret", "load_secret", "secret_fingerprint", "create_and_save"]
