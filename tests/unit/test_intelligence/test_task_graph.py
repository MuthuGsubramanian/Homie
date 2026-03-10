from datetime import datetime, timezone, timedelta

from homie_core.intelligence.task_graph import TaskGraph, TaskNode


def _ts(minutes_ago: int) -> str:
    dt = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.isoformat()


def test_new_observation_creates_task():
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(0))
    tasks = tg.get_tasks()
    assert len(tasks) == 1
    assert tasks[0].state == "active"
    assert "Code.exe" in tasks[0].apps


def test_related_observation_joins_existing_task():
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(2))
    tg.observe(process="Code.exe", title="config.py - Homie", timestamp=_ts(1))
    tasks = tg.get_tasks()
    assert len(tasks) == 1
    assert len(tasks[0].windows) == 2


def test_unrelated_observation_creates_new_task():
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(10))
    tg.observe(process="spotify.exe", title="Spotify - Now Playing", timestamp=_ts(3))
    tasks = tg.get_tasks()
    assert len(tasks) == 2


def test_task_pauses_after_inactivity():
    tg = TaskGraph(boundary_minutes=5)
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(20))
    tg.observe(process="chrome.exe", title="Google", timestamp=_ts(10))
    tasks = tg.get_tasks()
    code_task = [t for t in tasks if "Code.exe" in t.apps][0]
    assert code_task.state == "paused"


def test_task_resumes_when_revisited():
    tg = TaskGraph(boundary_minutes=5)
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(20))
    tg.observe(process="chrome.exe", title="Google", timestamp=_ts(10))
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(0))
    tasks = tg.get_tasks()
    code_task = [t for t in tasks if "Code.exe" in t.apps][0]
    assert code_task.state == "active"


def test_stuck_detection():
    tg = TaskGraph(boundary_minutes=5, stuck_minutes=15)
    for i in range(20, 0, -1):
        proc = "Code.exe" if i % 2 == 0 else "chrome.exe"
        tg.observe(process=proc, title="engine.py - Homie", timestamp=_ts(i))
    tasks = tg.get_tasks()
    assert any(t.state == "stuck" for t in tasks)


def test_serialize_and_restore():
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(5))
    data = tg.serialize()
    tg2 = TaskGraph.deserialize(data)
    assert len(tg2.get_tasks()) == 1
    assert tg2.get_tasks()[0].apps == tg.get_tasks()[0].apps


def test_get_incomplete_tasks():
    tg = TaskGraph(boundary_minutes=5)
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(20))
    tg.observe(process="chrome.exe", title="Google", timestamp=_ts(10))
    incomplete = tg.get_incomplete_tasks()
    assert len(incomplete) >= 1


def test_summary_generation():
    tg = TaskGraph()
    tg.observe(process="Code.exe", title="engine.py - Homie", timestamp=_ts(5))
    tg.observe(process="Code.exe", title="config.py - Homie", timestamp=_ts(3))
    summary = tg.summarize()
    assert "Code.exe" in summary
    assert isinstance(summary, str)
