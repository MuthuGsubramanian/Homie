from homie_core.memory.working import WorkingMemory


def test_update_and_get_snapshot():
    wm = WorkingMemory(max_age_seconds=300)
    wm.update("active_app", "VS Code")
    wm.update("active_file", "main.py")
    snap = wm.snapshot()
    assert snap["active_app"] == "VS Code"
    assert snap["active_file"] == "main.py"


def test_conversation_buffer():
    wm = WorkingMemory()
    wm.add_message("user", "hello")
    wm.add_message("assistant", "hi there")
    msgs = wm.get_conversation()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"


def test_conversation_max_length():
    wm = WorkingMemory(max_conversation_turns=3)
    for i in range(5):
        wm.add_message("user", f"msg {i}")
    msgs = wm.get_conversation()
    assert len(msgs) == 3
    assert msgs[0]["content"] == "msg 2"


def test_clear():
    wm = WorkingMemory()
    wm.update("key", "val")
    wm.add_message("user", "hi")
    wm.clear()
    assert wm.snapshot() == {}
    assert wm.get_conversation() == []
