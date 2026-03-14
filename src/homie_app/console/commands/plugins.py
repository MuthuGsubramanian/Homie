"""Handler for /plugins slash command — plugin management."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_plugins_list(args: str, **ctx) -> str:
    try:
        from homie_core.plugins.manager import PluginManager
        mgr = PluginManager()
        plugins = mgr.list_plugins()
        if not plugins:
            return "No plugins registered."
        lines = ["**Plugins:**"]
        for p in plugins:
            status = "enabled" if p["enabled"] else "disabled"
            lines.append(f"  {p['name']} ({status}) - {p['description']}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not list plugins: {e}"


def _handle_plugins_enable(args: str, **ctx) -> str:
    name = args.strip()
    if not name:
        return "Usage: /plugins enable <name>"
    try:
        from homie_core.plugins.manager import PluginManager
        mgr = PluginManager()
        if mgr.enable(name):
            return f"Enabled plugin '{name}'"
        return f"Failed to enable plugin '{name}'"
    except Exception as e:
        return f"Could not enable plugin: {e}"


def _handle_plugins_disable(args: str, **ctx) -> str:
    name = args.strip()
    if not name:
        return "Usage: /plugins disable <name>"
    try:
        from homie_core.plugins.manager import PluginManager
        mgr = PluginManager()
        if mgr.disable(name):
            return f"Disabled plugin '{name}'"
        return f"Failed to disable plugin '{name}'"
    except Exception as e:
        return f"Could not disable plugin: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="plugins",
        description="Plugin management (list, enable, disable)",
        args_spec="list|enable|disable",
        subcommands={
            "list": SlashCommand(name="list", description="List all plugins", handler_fn=_handle_plugins_list),
            "enable": SlashCommand(name="enable", description="Enable a plugin", args_spec="<name>", handler_fn=_handle_plugins_enable),
            "disable": SlashCommand(name="disable", description="Disable a plugin", args_spec="<name>", handler_fn=_handle_plugins_disable),
        },
    ))
