from homie_core.context.screen_monitor import ScreenMonitor, WindowInfo


def test_get_active_window():
    monitor = ScreenMonitor()
    info = monitor.get_active_window()
    assert isinstance(info, WindowInfo)
    assert info.timestamp != ""


def test_has_changed_first_call():
    monitor = ScreenMonitor()
    info = WindowInfo(title="VS Code", process_name="code.exe", pid=1234)
    assert monitor.has_changed(info) is True


def test_has_changed_same_window():
    monitor = ScreenMonitor()
    monitor._last_window = WindowInfo(title="VS Code", process_name="code.exe", pid=1234)
    same = WindowInfo(title="VS Code", process_name="code.exe", pid=1234)
    assert monitor.has_changed(same) is False


def test_has_changed_different_window():
    monitor = ScreenMonitor()
    monitor._last_window = WindowInfo(title="VS Code", process_name="code.exe", pid=1234)
    diff = WindowInfo(title="Chrome", process_name="chrome.exe", pid=5678)
    assert monitor.has_changed(diff) is True
