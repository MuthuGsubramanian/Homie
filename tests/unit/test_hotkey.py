from unittest.mock import MagicMock, patch

from homie_app.hotkey import HotkeyListener


def test_hotkey_listener_init():
    callback = MagicMock()
    listener = HotkeyListener(hotkey="alt+8", callback=callback)
    assert listener._hotkey == "alt+8"
    assert listener._callback is callback
    assert not listener._running


def test_hotkey_listener_start_stop():
    callback = MagicMock()
    listener = HotkeyListener(hotkey="alt+8", callback=callback)
    with patch("homie_app.hotkey.keyboard") as mock_kb:
        mock_listener = MagicMock()
        mock_kb.GlobalHotKeys.return_value = mock_listener
        listener.start()
        assert listener._running
        mock_listener.start.assert_called_once()

        listener.stop()
        assert not listener._running
        mock_listener.stop.assert_called_once()


def test_hotkey_triggers_callback():
    callback = MagicMock()
    listener = HotkeyListener(hotkey="alt+8", callback=callback)
    listener._on_activate()
    callback.assert_called_once()
