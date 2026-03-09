from datetime import timedelta
from homie_core.context.app_tracker import AppTracker
from homie_core.utils import utc_now


def test_track_single_app():
    tracker = AppTracker()
    tracker.track("VS Code")
    usage = tracker.get_usage()
    assert "VS Code" in usage


def test_track_switch():
    tracker = AppTracker()
    tracker.track("VS Code")
    tracker.track("Chrome")
    assert len(tracker._switches) == 1
    assert tracker._switches[0]["from"] == "VS Code"
    assert tracker._switches[0]["to"] == "Chrome"


def test_deep_work_detection():
    tracker = AppTracker()
    tracker._current_app = "VS Code"
    tracker._current_start = utc_now() - timedelta(minutes=50)
    assert tracker.is_deep_work(threshold_minutes=45) is True


def test_not_deep_work():
    tracker = AppTracker()
    tracker._current_app = "VS Code"
    tracker._current_start = utc_now() - timedelta(minutes=10)
    assert tracker.is_deep_work(threshold_minutes=45) is False


def test_get_top_apps():
    tracker = AppTracker()
    tracker._usage["VS Code"] = 3600
    tracker._usage["Chrome"] = 1800
    tracker._usage["Slack"] = 900
    top = tracker.get_top_apps(n=2)
    assert len(top) == 2
    assert top[0][0] == "VS Code"
