"""Tests for /settings slash command."""
import pytest
from unittest.mock import MagicMock, patch
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.settings import register


def test_settings_no_args_shows_menu():
    router = SlashCommandRouter()
    register(router, {})
    with patch("builtins.input", return_value="10"):
        result = router.dispatch("/settings", **{"config": MagicMock(), "_router": router})
    # Should return empty string after selecting "Back"
    assert result == "" or "settings" in result.lower()


def test_settings_unknown_category():
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/settings nonexistent", **{"config": MagicMock(), "_router": router})
    assert "Unknown category" in result


def test_settings_no_config():
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/settings", **{"_router": router})
    assert "No configuration" in result
