"""Tests for autocomplete integration."""
import pytest
from homie_app.console.autocomplete import HomieCompleter
from homie_app.console.router import SlashCommandRouter, SlashCommand


def test_completer_returns_slash_commands():
    router = SlashCommandRouter()
    router.register(SlashCommand(name="connect", description="Connect"))
    router.register(SlashCommand(name="connections", description="List"))
    router.register(SlashCommand(name="help", description="Help"))

    completer = HomieCompleter(router)
    from prompt_toolkit.document import Document
    doc = Document("/con", cursor_position=4)
    completions = list(completer.get_completions(doc, None))
    texts = [c.text for c in completions]
    assert "connect" in texts or "/connect" in texts
    assert "help" not in texts


def test_completer_empty_slash():
    router = SlashCommandRouter()
    router.register(SlashCommand(name="help", description="Help"))
    router.register(SlashCommand(name="quit", description="Quit"))

    completer = HomieCompleter(router)
    from prompt_toolkit.document import Document
    doc = Document("/", cursor_position=1)
    completions = list(completer.get_completions(doc, None))
    texts = [c.text for c in completions]
    assert "help" in texts
    assert "quit" in texts


def test_completer_no_slash_no_completions():
    router = SlashCommandRouter()
    router.register(SlashCommand(name="help", description="Help"))

    completer = HomieCompleter(router)
    from prompt_toolkit.document import Document
    doc = Document("hello", cursor_position=5)
    completions = list(completer.get_completions(doc, None))
    assert len(completions) == 0


def test_completer_subcommands():
    router = SlashCommandRouter()
    sub = SlashCommand(name="start", description="Start daemon", handler_fn=lambda **k: "")
    cmd = SlashCommand(name="daemon", description="Daemon management", subcommands={"start": sub})
    router.register(cmd)

    completer = HomieCompleter(router)
    from prompt_toolkit.document import Document
    doc = Document("/daemon s", cursor_position=9)
    completions = list(completer.get_completions(doc, None))
    texts = [c.text for c in completions]
    assert "start" in texts
