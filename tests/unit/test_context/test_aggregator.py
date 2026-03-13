from unittest.mock import MagicMock, patch
from homie_core.context.aggregator import ContextAggregator
from homie_core.context.screen_monitor import WindowInfo
from homie_core.context.app_tracker import AppTracker
from homie_core.context.clipboard import ClipboardMonitor
from homie_core.memory.working import WorkingMemory


def test_tick_updates_working_memory():
    wm = WorkingMemory()
    screen = MagicMock()
    screen.get_active_window.return_value = WindowInfo(title="VS Code - main.py", process_name="code.exe", pid=1234, timestamp="t")
    agg = ContextAggregator(working_memory=wm, screen_monitor=screen)
    snapshot = agg.tick()
    assert snapshot["active_window"] == "VS Code - main.py"
    assert wm.get("active_window") == "VS Code - main.py"


def test_get_full_context():
    wm = WorkingMemory()
    wm.update("test", "value")
    agg = ContextAggregator(working_memory=wm)
    ctx = agg.get_full_context()
    assert "working_memory" in ctx
    assert "top_apps" in ctx
    assert "is_deep_work" in ctx
