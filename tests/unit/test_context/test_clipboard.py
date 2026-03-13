from homie_core.context.clipboard import ClipboardMonitor


def test_history_starts_empty():
    cm = ClipboardMonitor()
    assert cm.get_history() == []


def test_manual_add_to_history():
    cm = ClipboardMonitor()
    cm._last_content = None
    # Simulate clipboard change
    cm._history.append({"content": "hello world", "timestamp": "2024-01-01T00:00:00"})
    cm._history.append({"content": "foo bar", "timestamp": "2024-01-01T00:01:00"})
    history = cm.get_history(n=5)
    assert len(history) == 2


def test_search():
    cm = ClipboardMonitor()
    cm._history.append({"content": "hello world", "timestamp": "t1"})
    cm._history.append({"content": "foo bar", "timestamp": "t2"})
    cm._history.append({"content": "hello again", "timestamp": "t3"})
    results = cm.search("hello")
    assert len(results) == 2


def test_max_history():
    cm = ClipboardMonitor(max_history=3)
    for i in range(5):
        cm._history.append({"content": f"item {i}", "timestamp": f"t{i}"})
    assert len(cm._history) == 3
