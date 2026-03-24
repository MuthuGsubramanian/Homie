"""Handler for /desktop slash command — launch desktop companion."""
from __future__ import annotations

from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_desktop(args: str, **ctx) -> str:
    from homie_app.desktop import DesktopCompanion
    companion = DesktopCompanion()
    companion.start()
    return "Desktop companion stopped."


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="desktop",
        description="Launch the Homie desktop companion (system tray + morning briefing)",
        handler_fn=_handle_desktop,
    ))
