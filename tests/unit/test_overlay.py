from unittest.mock import MagicMock, patch

from homie_app.overlay import OverlayPopup


def test_overlay_init():
    callback = MagicMock()
    overlay = OverlayPopup(on_submit=callback)
    assert overlay._on_submit is callback
    assert not overlay._visible


def test_overlay_submit_calls_callback():
    callback = MagicMock(return_value="Hello back!")
    overlay = OverlayPopup(on_submit=callback)
    result = overlay._handle_submit("Hello")
    callback.assert_called_once_with("Hello")
    assert result == "Hello back!"


def test_overlay_toggle():
    overlay = OverlayPopup(on_submit=MagicMock())
    assert not overlay._visible
    overlay._visible = True
    assert overlay._visible
    overlay._visible = False
    assert not overlay._visible


def test_overlay_empty_submit_ignored():
    callback = MagicMock()
    overlay = OverlayPopup(on_submit=callback)
    result = overlay._handle_submit("")
    callback.assert_not_called()
    assert result is None

    result = overlay._handle_submit("   ")
    callback.assert_not_called()
