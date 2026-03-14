"""Tests for /email, /consent-log, /vault slash commands."""
import pytest
from unittest.mock import MagicMock
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.email import register as reg_email
from homie_app.console.commands.consent import register as reg_consent
from homie_app.console.commands.vault import register as reg_vault


def test_email_no_subcommand_shows_help():
    router = SlashCommandRouter()
    ctx = {"_router": router}
    reg_email(router, ctx)
    result = router.dispatch("/email", **ctx)
    assert "summary" in result
    assert "sync" in result


def test_consent_log_no_provider():
    router = SlashCommandRouter()
    ctx = {"_router": router}
    reg_consent(router, ctx)
    result = router.dispatch("/consent-log", **ctx)
    assert "usage" in result.lower() or "provider" in result.lower()


def test_vault_status():
    router = SlashCommandRouter()
    vault = MagicMock()
    vault.get_all_connections.return_value = []
    vault.has_password = True
    ctx = {"vault": vault, "_router": router}
    reg_vault(router, ctx)
    result = router.dispatch("/vault", **ctx)
    assert "vault" in result.lower()
    assert "0 active" in result


def test_consent_log_with_provider():
    router = SlashCommandRouter()
    vault = MagicMock()
    entry = MagicMock()
    entry.timestamp = 1710000000.0
    entry.action = "connected"
    vault.get_consent_history.return_value = [entry]
    ctx = {"vault": vault, "_router": router}
    reg_consent(router, ctx)
    result = router.dispatch("/consent-log gmail", **ctx)
    assert "gmail" in result
    assert "connected" in result
