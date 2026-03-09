import pytest
from homie_core.plugins.base import PluginResult

from homie_app.plugins.system_plugin import SystemPlugin
from homie_app.plugins.clipboard_plugin import ClipboardPlugin
from homie_app.plugins.health_plugin import HealthPlugin
from homie_app.plugins.git_plugin import GitPlugin
from homie_app.plugins.shortcuts_plugin import ShortcutsPlugin
from homie_app.plugins.terminal_plugin import TerminalPlugin
from homie_app.plugins.network_plugin import NetworkPlugin
from homie_app.plugins.workflows_plugin import WorkflowsPlugin
from homie_app.plugins.notes_plugin import NotesPlugin
from homie_app.plugins.browser_plugin import BrowserPlugin
from homie_app.plugins.ide_plugin import IDEPlugin
from homie_app.plugins.music_plugin import MusicPlugin


def test_system_plugin_context():
    p = SystemPlugin()
    p.on_activate({})
    ctx = p.on_context()
    assert "cpu_percent" in ctx
    assert "ram_percent" in ctx


def test_system_plugin_query_status():
    p = SystemPlugin()
    p.on_activate({})
    result = p.on_query("status", {})
    assert result.success is True
    assert "cpu" in result.data


def test_system_plugin_query_processes():
    p = SystemPlugin()
    p.on_activate({})
    result = p.on_query("processes", {})
    assert result.success is True
    assert isinstance(result.data, list)


def test_clipboard_plugin():
    p = ClipboardPlugin()
    p.on_activate({})
    result = p.on_query("history", {})
    assert result.success is True


def test_health_plugin_context():
    p = HealthPlugin()
    p.on_activate({"break_interval": 60})
    ctx = p.on_context()
    assert "minutes_since_break" in ctx


def test_health_plugin_record_break():
    p = HealthPlugin()
    p.on_activate({})
    result = p.on_action("record_break", {})
    assert result.success is True


def test_git_plugin_status():
    p = GitPlugin()
    p.on_activate({"repo_path": "."})
    result = p.on_query("status", {})
    assert result.success is True


def test_git_plugin_branch():
    p = GitPlugin()
    p.on_activate({"repo_path": "."})
    result = p.on_query("branch", {})
    assert result.success is True


def test_shortcuts_plugin_add_remove():
    p = ShortcutsPlugin()
    p.on_activate({})
    result = p.on_action("add", {"name": "test", "trigger": "hello", "response": "world"})
    assert result.success is True
    result = p.on_query("list", {})
    assert result.success is True
    assert len(result.data) == 1
    p.on_action("remove", {"name": "test"})
    result = p.on_query("list", {})
    assert len(result.data) == 0


def test_terminal_plugin():
    p = TerminalPlugin()
    p.on_activate({})
    result = p.on_query("history", {"n": 5})
    assert result.success is True


def test_network_plugin_context():
    p = NetworkPlugin()
    p.on_activate({})
    ctx = p.on_context()
    assert "bytes_sent" in ctx


def test_network_plugin_status():
    p = NetworkPlugin()
    p.on_activate({})
    result = p.on_query("status", {})
    assert result.success is True


def test_workflows_plugin():
    p = WorkflowsPlugin()
    p.on_activate({})
    result = p.on_action("add", {"name": "morning", "steps": ["check email", "review calendar"]})
    assert result.success is True
    result = p.on_query("list", {})
    assert "morning" in result.data


def test_notes_plugin():
    p = NotesPlugin()
    p.on_activate({})
    result = p.on_query("recent", {"n": 5})
    assert result.success is True


def test_browser_plugin():
    p = BrowserPlugin()
    p.on_activate({})
    result = p.on_query("recent_history", {"limit": 5})
    assert result.success is True


def test_ide_plugin():
    p = IDEPlugin()
    p.on_activate({})
    result = p.on_query("recent_projects", {})
    assert result.success is True


def test_music_plugin():
    p = MusicPlugin()
    p.on_activate({})
    result = p.on_query("now_playing", {})
    assert result.success is True


def test_all_plugins_have_name():
    plugins = [SystemPlugin, ClipboardPlugin, HealthPlugin, GitPlugin,
               ShortcutsPlugin, TerminalPlugin, NetworkPlugin, WorkflowsPlugin,
               NotesPlugin, BrowserPlugin, IDEPlugin, MusicPlugin]
    for cls in plugins:
        p = cls()
        assert p.name != "", f"{cls.__name__} has no name"
        assert p.description != "", f"{cls.__name__} has no description"
