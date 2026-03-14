"""Tests for remaining migrated slash commands."""
import pytest
from unittest.mock import MagicMock, patch
from homie_app.console.router import SlashCommandRouter


def test_daemon_no_subcommand():
    from homie_app.console.commands.daemon import register
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/daemon", **{"_router": router})
    assert "start" in result
    assert "stop" in result
    assert "status" in result


def test_daemon_status_no_pid():
    from homie_app.console.commands.daemon import register
    router = SlashCommandRouter()
    register(router, {})
    with patch("pathlib.Path.exists", return_value=False):
        result = router.dispatch("/daemon status", **{"_router": router})
    assert "not running" in result


def test_skills_no_skills():
    from homie_app.console.commands.skills import register
    router = SlashCommandRouter()
    register(router, {})
    with patch("homie_core.skills.loader.SkillLoader") as MockLoader:
        MockLoader.return_value.scan.return_value = []
        result = router.dispatch("/skills", **{"_router": router})
    assert "no skills" in result.lower()


def test_schedule_no_subcommand():
    from homie_app.console.commands.schedule import register
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/schedule", **{"_router": router})
    assert "add" in result
    assert "list" in result


def test_insights_command():
    from homie_app.console.commands.insights import register
    router = SlashCommandRouter()
    register(router, {})
    with patch("homie_core.analytics.insights.InsightsEngine") as MockEngine:
        mock_engine = MockEngine.return_value
        mock_engine.generate_insights.return_value = {}
        mock_engine.format_terminal.return_value = "Insights: 10 sessions"
        result = router.dispatch("/insights", **{"config": MagicMock(storage=MagicMock(path="/tmp")), "_router": router})
    assert "10 sessions" in result


def test_model_no_subcommand():
    from homie_app.console.commands.model import register
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/model", **{"_router": router})
    assert "list" in result


def test_plugins_no_subcommand():
    from homie_app.console.commands.plugins import register
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/plugins", **{"_router": router})
    assert "list" in result or "enable" in result


def test_folder_no_subcommand():
    from homie_app.console.commands.folder import register
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/folder", **{"_router": router})
    assert "watch" in result or "list" in result


def test_backup_no_args():
    from homie_app.console.commands.backup import register
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/backup", **{"_router": router})
    assert "path" in result.lower() or "usage" in result.lower()


def test_voice_no_subcommand():
    from homie_app.console.commands.voice import register
    router = SlashCommandRouter()
    register(router, {})
    result = router.dispatch("/voice", **{"_router": router})
    assert "status" in result or "enable" in result
