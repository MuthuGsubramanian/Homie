from homie_app.hotkey import _HOTKEY_MAP


def test_ctrl_8_in_hotkey_map():
    assert "ctrl+8" in _HOTKEY_MAP
    assert _HOTKEY_MAP["ctrl+8"] == "<ctrl>+8"
