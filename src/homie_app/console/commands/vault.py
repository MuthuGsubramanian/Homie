"""Handler for /vault slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_vault(args: str, **ctx) -> str:
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        connections = vault.get_all_connections()
        active = sum(1 for c in connections if c.connected)
        has_pw = vault.has_password
        return (
            f"**Vault Status:**\n"
            f"  Connections: {active} active / {len(connections)} total\n"
            f"  Password: {'set' if has_pw else 'not set'}"
        )
    except Exception as e:
        return f"Could not check vault: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="vault",
        description="Vault health and status",
        handler_fn=_handle_vault,
    ))
