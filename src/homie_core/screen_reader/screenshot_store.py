from __future__ import annotations
import hashlib
import time
from pathlib import Path
from typing import Optional


class ScreenshotStore:
    """Rolling screenshot store with JPEG compression and retention policy."""
    QUALITY = 50
    MAX_HEIGHT = 540

    def __init__(self, base_dir: str | Path, retention_days: int = 7):
        self._dir = Path(base_dir) / "screenshots"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days

    def save(self, image_bytes: bytes, window_title: str = "") -> Optional[Path]:
        try:
            from PIL import Image
            from io import BytesIO
            img = Image.open(BytesIO(image_bytes))
            if img.height > self.MAX_HEIGHT:
                ratio = self.MAX_HEIGHT / img.height
                img = img.resize((int(img.width * ratio), self.MAX_HEIGHT), Image.LANCZOS)
            ts = int(time.time())
            slug = hashlib.md5(window_title.encode()).hexdigest()[:8]
            path = self._dir / f"{ts}_{slug}.jpg"
            img.save(path, format="JPEG", quality=self.QUALITY)
            return path
        except Exception:
            return None

    def purge_old(self) -> int:
        cutoff = time.time() - (self._retention_days * 86400)
        removed = 0
        for f in self._dir.glob("*.jpg"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
        return removed

    def count(self) -> int:
        return len(list(self._dir.glob("*.jpg")))

    def size_mb(self) -> float:
        return sum(f.stat().st_size for f in self._dir.glob("*.jpg")) / (1024**2)
