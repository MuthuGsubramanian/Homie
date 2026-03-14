"""Tests for SlashCommandRouter — registration, dispatch, subcommands."""
import pytest
from homie_app.console.router import SlashCommand, SlashCommandRouter


def test_register_and_dispatch():
    router = SlashCommandRouter()
    called_with = {}

    def handler(args: str, **ctx):
        called_with["args"] = args
        return "ok"

    router.register(SlashCommand(
        name="test",
        description="A test command",
        handler_fn=handler,
    ))
    result = router.dispatch("/test some args", **{})
    assert result == "ok"
    assert called_with["args"] == "some args"


def test_dispatch_unknown_command():
    router = SlashCommandRouter()
    result = router.dispatch("/nonexistent", **{})
    assert "Unknown command" in result
    assert "/help" in result


def test_list_commands():
    router = SlashCommandRouter()
    router.register(SlashCommand(name="alpha", description="First"))
    router.register(SlashCommand(name="beta", description="Second"))
    commands = router.list_commands()
    assert len(commands) == 2
    assert commands[0].name == "alpha"


def test_dispatch_with_subcommands():
    router = SlashCommandRouter()
    sub_called = {}

    def sub_handler(args: str, **ctx):
        sub_called["args"] = args
        return "sub ok"

    def parent_handler(args: str, **ctx):
        return "parent ok"

    router.register(SlashCommand(
        name="daemon",
        description="Manage daemon",
        handler_fn=parent_handler,
        subcommands={
            "start": SlashCommand(name="start", description="Start daemon", handler_fn=sub_handler),
        },
    ))
    result = router.dispatch("/daemon start extra", **{})
    assert result == "sub ok"
    assert sub_called["args"] == "extra"

    result = router.dispatch("/daemon", **{})
    assert "start" in result


def test_autocomplete_matching():
    router = SlashCommandRouter()
    router.register(SlashCommand(name="connect", description="Connect provider"))
    router.register(SlashCommand(name="connections", description="List connections"))
    router.register(SlashCommand(name="consent-log", description="Consent log"))
    router.register(SlashCommand(name="help", description="Help"))

    matches = router.get_completions("con")
    names = [m.name for m in matches]
    assert "connect" in names
    assert "connections" in names
    assert "consent-log" in names
    assert "help" not in names


def test_bare_slash_lists_all():
    router = SlashCommandRouter()
    router.register(SlashCommand(name="help", description="Help"))
    router.register(SlashCommand(name="quit", description="Quit"))
    result = router.dispatch("/", **{})
    assert "help" in result
    assert "quit" in result
