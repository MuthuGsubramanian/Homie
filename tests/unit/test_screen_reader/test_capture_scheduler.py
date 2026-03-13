import time
from unittest.mock import MagicMock, patch
from homie_core.screen_reader.capture_scheduler import CaptureScheduler
from homie_core.config import ScreenReaderConfig


class TestCaptureScheduler:
    def test_init(self):
        cfg = ScreenReaderConfig(enabled=True, level=1)
        sched = CaptureScheduler(config=cfg)
        assert sched is not None

    def test_disabled_does_nothing(self):
        cfg = ScreenReaderConfig(enabled=False)
        sched = CaptureScheduler(config=cfg)
        sched.tick()
        # No error, no capture

    def test_level1_only_tracks_windows(self):
        cfg = ScreenReaderConfig(enabled=True, level=1)
        tracker = MagicMock()
        sched = CaptureScheduler(config=cfg, window_tracker=tracker)
        sched.tick()
        tracker.get_active_window.assert_called_once()

    def test_dnd_pauses_capture(self):
        cfg = ScreenReaderConfig(enabled=True, level=1, dnd=True)
        tracker = MagicMock()
        sched = CaptureScheduler(config=cfg, window_tracker=tracker)
        sched.tick()
        tracker.get_active_window.assert_not_called()

    def test_event_driven_fires_on_window_change(self):
        cfg = ScreenReaderConfig(enabled=True, level=2, event_driven=True)
        tracker = MagicMock()
        ocr = MagicMock()
        ocr.extract_text.return_value = "some text"
        from homie_core.screen_reader.window_tracker import WindowInfo
        tracker.get_active_window.side_effect = [
            WindowInfo("Chrome", "chrome.exe", 1),
            WindowInfo("VS Code", "code.exe", 2),
        ]
        tracker.has_changed.return_value = True
        tracker.is_blocked.return_value = False
        sched = CaptureScheduler(config=cfg, window_tracker=tracker, ocr_reader=ocr)
        sched.tick()  # first capture
        sched.tick()  # window changed -> triggers OCR
        assert ocr.capture_screen.called

    def test_blocked_window_skips_capture(self):
        cfg = ScreenReaderConfig(enabled=True, level=2)
        tracker = MagicMock()
        ocr = MagicMock()
        from homie_core.screen_reader.window_tracker import WindowInfo
        tracker.get_active_window.return_value = WindowInfo("1Password", "1password.exe", 1)
        tracker.is_blocked.return_value = True
        sched = CaptureScheduler(config=cfg, window_tracker=tracker, ocr_reader=ocr)
        sched.tick()
        ocr.capture_screen.assert_not_called()
