"""Tests for ScreenshotStore."""
from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _make_jpeg_bytes() -> bytes:
    """Return minimal valid JPEG bytes via Pillow (skips if unavailable)."""
    try:
        from PIL import Image
        buf = BytesIO()
        img = Image.new("RGB", (100, 80), color=(255, 0, 0))
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        pytest.skip("Pillow not available")


def _make_tall_jpeg_bytes(width: int = 100, height: int = 1080) -> bytes:
    """Return JPEG bytes for a tall image."""
    from PIL import Image
    buf = BytesIO()
    img = Image.new("RGB", (width, height), color=(0, 128, 0))
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestScreenshotStore:
    def test_save_returns_path(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        store = ScreenshotStore(tmp_path, retention_days=7)
        data = _make_jpeg_bytes()
        result = store.save(data, window_title="Test Window")
        assert result is not None
        assert result.exists()
        assert result.suffix == ".jpg"

    def test_save_creates_screenshots_subdirectory(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        store = ScreenshotStore(tmp_path, retention_days=7)
        data = _make_jpeg_bytes()
        store.save(data)
        assert (tmp_path / "screenshots").is_dir()

    def test_save_resizes_tall_image(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        from PIL import Image
        store = ScreenshotStore(tmp_path)
        data = _make_tall_jpeg_bytes(width=100, height=1080)
        path = store.save(data, window_title="tall")
        assert path is not None
        img = Image.open(path)
        assert img.height <= ScreenshotStore.MAX_HEIGHT

    def test_save_does_not_resize_short_image(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        from PIL import Image
        store = ScreenshotStore(tmp_path)
        data = _make_jpeg_bytes()  # 100x80
        path = store.save(data, window_title="small")
        assert path is not None
        img = Image.open(path)
        assert img.height == 80

    def test_count_reflects_saved_files(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        store = ScreenshotStore(tmp_path)
        assert store.count() == 0
        data = _make_jpeg_bytes()
        store.save(data, "win1")
        store.save(data, "win2")
        assert store.count() == 2

    def test_size_mb_is_positive_after_save(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        store = ScreenshotStore(tmp_path)
        data = _make_jpeg_bytes()
        store.save(data, "win")
        assert store.size_mb() > 0.0

    def test_size_mb_zero_when_empty(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        store = ScreenshotStore(tmp_path)
        assert store.size_mb() == 0.0

    def test_purge_old_removes_expired_files(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        store = ScreenshotStore(tmp_path, retention_days=1)
        # Manually plant a stale .jpg
        stale = tmp_path / "screenshots" / "0000000000_aabbccdd.jpg"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"fake")
        # Set mtime to 2 days ago
        old_time = time.time() - 2 * 86400
        import os
        os.utime(stale, (old_time, old_time))
        removed = store.purge_old()
        assert removed == 1
        assert not stale.exists()

    def test_purge_old_keeps_recent_files(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        store = ScreenshotStore(tmp_path, retention_days=7)
        data = _make_jpeg_bytes()
        path = store.save(data, "recent")
        assert path is not None
        removed = store.purge_old()
        assert removed == 0
        assert path.exists()

    def test_purge_old_returns_zero_when_empty(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        store = ScreenshotStore(tmp_path)
        assert store.purge_old() == 0

    def test_save_returns_none_on_invalid_bytes(self, tmp_path):
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        store = ScreenshotStore(tmp_path)
        result = store.save(b"not an image", window_title="broken")
        assert result is None

    def test_save_handles_missing_pil_gracefully(self, tmp_path):
        """When PIL is not importable, save() should return None, not raise."""
        from homie_core.screen_reader.screenshot_store import ScreenshotStore
        store = ScreenshotStore(tmp_path)
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("PIL mocked as missing")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = store.save(b"bytes", window_title="test")
        assert result is None
