import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from homie_core.intelligence.session_tracker import SessionTracker
from homie_core.intelligence.task_graph import TaskGraph


def _ts(minutes_ago: int) -> str:
    dt = datetime(2026, 3, 10, 18, 0, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.isoformat()


def test_save_and_load_session(tmp_path):
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(10))
    tg.observe(process="chrome.exe", title="Stack Overflow", timestamp=_ts(5))

    tracker = SessionTracker(storage_dir=tmp_path)
    tracker.save_session(tg, apps_used={"Code.exe": 1200.0, "chrome.exe": 300.0})

    loaded = tracker.load_last_session()
    assert loaded is not None
    assert loaded["task_graph"] is not None
    assert "Code.exe" in loaded["apps_used"]


def test_generate_resumption_summary(tmp_path):
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(10))

    tracker = SessionTracker(storage_dir=tmp_path)
    tracker.save_session(tg, apps_used={"Code.exe": 3600.0})

    summary = tracker.get_resumption_summary()
    assert summary is not None
    assert "Code.exe" in summary


def test_no_session_returns_none(tmp_path):
    tracker = SessionTracker(storage_dir=tmp_path)
    assert tracker.load_last_session() is None
    assert tracker.get_resumption_summary() is None


def test_generate_end_of_day_digest(tmp_path):
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(120))
    tg.observe(process="chrome.exe", title="Google", timestamp=_ts(60))
    tg.observe(process="Code.exe", title="config.py - Homie", timestamp=_ts(30))

    tracker = SessionTracker(storage_dir=tmp_path)
    digest = tracker.generate_digest(
        tg,
        apps_used={"Code.exe": 5400.0, "chrome.exe": 1800.0},
        switch_count=15,
    )
    assert "Code.exe" in digest
    assert isinstance(digest, str)
