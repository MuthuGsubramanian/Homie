from datetime import datetime, timezone, timedelta
from pathlib import Path

from homie_core.intelligence.briefing import BriefingGenerator
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.intelligence.session_tracker import SessionTracker


def _ts(minutes_ago: int) -> str:
    dt = datetime(2026, 3, 10, 18, 0, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.isoformat()


def test_morning_briefing_with_previous_session(tmp_path):
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(120))
    tg.observe(process="chrome.exe", title="Google", timestamp=_ts(60))

    tracker = SessionTracker(storage_dir=tmp_path)
    tracker.save_session(tg, apps_used={"Code.exe": 5400.0, "chrome.exe": 1200.0})

    gen = BriefingGenerator(session_tracker=tracker, user_name="Master")
    briefing = gen.morning_briefing()

    assert "Master" in briefing or "Good" in briefing
    assert "Code.exe" in briefing


def test_morning_briefing_without_previous_session(tmp_path):
    tracker = SessionTracker(storage_dir=tmp_path)
    gen = BriefingGenerator(session_tracker=tracker, user_name="Master")
    briefing = gen.morning_briefing()
    assert "Master" in briefing or "Good" in briefing
    assert isinstance(briefing, str)


def test_end_of_day_digest(tmp_path):
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(120))

    tracker = SessionTracker(storage_dir=tmp_path)
    gen = BriefingGenerator(session_tracker=tracker, user_name="Master")
    digest = gen.end_of_day_digest(
        task_graph=tg,
        apps_used={"Code.exe": 7200.0},
        switch_count=10,
    )
    assert "Code.exe" in digest
    assert isinstance(digest, str)


def test_greeting_varies_by_hour():
    from homie_core.intelligence.briefing import _greeting
    assert "morning" in _greeting(8).lower()
    assert "afternoon" in _greeting(14).lower()
    assert "evening" in _greeting(20).lower()
