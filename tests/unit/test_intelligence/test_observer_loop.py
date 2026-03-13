import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from homie_core.intelligence.observer_loop import ObserverLoop
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.context.screen_monitor import WindowInfo
from homie_core.memory.working import WorkingMemory


def test_observer_loop_init():
    wm = WorkingMemory()
    tg = TaskGraph()
    loop = ObserverLoop(working_memory=wm, task_graph=tg)
    assert not loop.is_running


def test_observer_processes_window_event():
    wm = WorkingMemory()
    tg = TaskGraph()
    pr_callback = MagicMock()
    loop = ObserverLoop(working_memory=wm, task_graph=tg,
                        on_context_change=pr_callback)

    window = WindowInfo(
        title="engine.py - Homie",
        process_name="Code.exe",
        pid=1234,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    loop._handle_window_change(window)

    assert wm.get("active_window") == "engine.py - Homie"
    assert len(tg.get_tasks()) == 1
    pr_callback.assert_called_once_with("Code.exe", "engine.py - Homie")


def test_observer_debounces_same_window():
    wm = WorkingMemory()
    tg = TaskGraph()
    pr_callback = MagicMock()
    loop = ObserverLoop(working_memory=wm, task_graph=tg,
                        on_context_change=pr_callback)

    window = WindowInfo(title="engine.py", process_name="Code.exe", pid=1,
                        timestamp=datetime.now(timezone.utc).isoformat())
    loop._handle_window_change(window)
    loop._handle_window_change(window)

    assert pr_callback.call_count == 1


def test_observer_cpu_budget():
    loop = ObserverLoop(working_memory=WorkingMemory(), task_graph=TaskGraph(),
                        cpu_budget=0.05)
    assert loop._cpu_budget == 0.05
