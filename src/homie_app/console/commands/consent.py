"""Handler for /consent-log slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_consent_log(args: str, **ctx) -> str:
    provider = args.strip()
    if not provider:
        return "Usage: /consent-log <provider>"
    try:
        from datetime import datetime
        from homie_core.vault.secure_vault import SecureVault

        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        history = vault.get_consent_history(provider)
        if not history:
            return f"No consent history for '{provider}'."
        lines = [f"**Consent log for {provider}:**"]
        for entry in history:
            dt = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  {dt}  {entry.action}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not check consent log: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="consent-log",
        description="Show consent audit trail (e.g., /consent-log gmail)",
        args_spec="<provider>",
        handler_fn=_handle_consent_log,
    ))
