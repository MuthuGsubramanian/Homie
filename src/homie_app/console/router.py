"""Slash command router — registry, dispatch, and autocomplete."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class SlashCommand:
    """A registered slash command."""
    name: str
    description: str
    args_spec: str = ""
    handler_fn: Optional[Callable[..., str]] = None
    subcommands: dict[str, "SlashCommand"] = field(default_factory=dict)
    autocomplete_fn: Optional[Callable[[str], list[str]]] = None

    def format_help(self) -> str:
        """One-line help: /name — description."""
        args_part = f" {self.args_spec}" if self.args_spec else ""
        return f"  /{self.name}{args_part:<{16 - len(self.name)}} — {self.description}"


class SlashCommandRouter:
    """Registry and dispatcher for slash commands."""

    def __init__(self):
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        self._commands[command.name] = command

    def list_commands(self) -> list[SlashCommand]:
        return sorted(self._commands.values(), key=lambda c: c.name)

    def get_completions(self, prefix: str) -> list[SlashCommand]:
        """Return commands whose name starts with prefix."""
        return [c for c in self._commands.values() if c.name.startswith(prefix)]

    def dispatch(self, raw_input: str, **ctx: Any) -> str:
        """Parse and dispatch a slash command. Returns response string."""
        text = raw_input.lstrip("/").strip()

        # Bare "/" — list all commands
        if not text:
            return self._format_command_list()

        parts = text.split(maxsplit=1)
        cmd_name = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        command = self._commands.get(cmd_name)
        if not command:
            return f"Unknown command: /{cmd_name}. Type /help to see available commands."

        # Check for subcommand dispatch
        if command.subcommands and rest:
            sub_parts = rest.split(maxsplit=1)
            sub_name = sub_parts[0].lower()
            if sub_name in command.subcommands:
                sub_rest = sub_parts[1] if len(sub_parts) > 1 else ""
                return command.subcommands[sub_name].handler_fn(args=sub_rest, **ctx)

        # No subcommand match — if command has subcommands and no rest, show help
        if command.subcommands and not rest:
            lines = [f"**/{command.name}** subcommands:"]
            for sc in sorted(command.subcommands.values(), key=lambda s: s.name):
                lines.append(sc.format_help())
            return "\n".join(lines)

        # Direct handler
        if command.handler_fn:
            return command.handler_fn(args=rest, **ctx)

        # Has subcommands but no direct handler and unrecognized sub
        if command.subcommands:
            lines = [f"**/{command.name}** subcommands:"]
            for sc in sorted(command.subcommands.values(), key=lambda s: s.name):
                lines.append(sc.format_help())
            return "\n".join(lines)

        return f"/{cmd_name}: no handler registered."

    def _format_command_list(self) -> str:
        lines = ["**Available Commands:**"]
        for cmd in self.list_commands():
            lines.append(cmd.format_help())
        return "\n".join(lines)
