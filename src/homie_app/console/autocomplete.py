"""Autocomplete for slash commands using prompt_toolkit."""
from __future__ import annotations

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from homie_app.console.router import SlashCommandRouter


class HomieCompleter(Completer):
    """Autocomplete slash commands."""

    def __init__(self, router: SlashCommandRouter):
        self._router = router

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        after_slash = text[1:]

        parts = after_slash.split(maxsplit=1)
        if len(parts) == 2:
            cmd_name = parts[0]
            sub_prefix = parts[1]
            cmd = self._router._commands.get(cmd_name)
            if cmd and cmd.subcommands:
                for name, sc in sorted(cmd.subcommands.items()):
                    if name.startswith(sub_prefix):
                        yield Completion(
                            name,
                            start_position=-len(sub_prefix),
                            display_meta=sc.description,
                        )
            return

        prefix = parts[0] if parts else ""
        for cmd in self._router.list_commands():
            if cmd.name.startswith(prefix):
                yield Completion(
                    cmd.name,
                    start_position=-len(prefix),
                    display_meta=cmd.description,
                )
