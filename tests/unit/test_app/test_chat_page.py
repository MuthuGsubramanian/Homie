"""Tests for chat page generator."""
from __future__ import annotations
import pytest


def test_render_chat_page():
    from homie_app.tray.chat_page import render_chat_page
    html = render_chat_page(session_token="tok-123", api_port=8721)
    assert "<html" in html
    assert "Homie" in html
    assert "tok-123" in html
    assert "8721" in html


def test_render_chat_page_has_input():
    from homie_app.tray.chat_page import render_chat_page
    html = render_chat_page(session_token="tok", api_port=8721)
    assert "input" in html.lower() or "textarea" in html.lower()
    assert "send" in html.lower()
