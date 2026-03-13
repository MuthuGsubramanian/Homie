"""Handler for /social slash command — social messaging."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _get_social_service(ctx):
    """Build SocialService from context config."""
    from pathlib import Path
    from homie_core.vault.secure_vault import SecureVault
    from homie_core.social import SocialService
    cfg = ctx.get("config")
    if not cfg:
        return None, None, "No configuration loaded."
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()
    service = SocialService(vault=vault)
    workspaces = service.initialize()
    if not workspaces:
        vault.lock()
        return None, vault, "No social platforms connected. Run: homie connect slack"
    return service, vault, None


def _handle_social_channels(args: str, **ctx) -> str:
    try:
        service, vault, err = _get_social_service(ctx)
        if err:
            return err
        try:
            channels = service.list_channels()
            if not channels:
                return "No channels found."
            lines = ["**Channels:**"]
            for ch in channels:
                private = " (private)" if ch.get("is_private") else ""
                lines.append(f"  {ch['id']}  #{ch['name']}{private}  ({ch.get('member_count', '?')} members)")
            return "\n".join(lines)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not list channels: {e}"


def _handle_social_recent(args: str, **ctx) -> str:
    parts = args.strip().split()
    if not parts:
        return "Usage: /social recent <channel_id> [--limit N]"
    try:
        channel = parts[0]
        limit = 20
        if "--limit" in parts:
            limit = int(parts[parts.index("--limit") + 1])

        service, vault, err = _get_social_service(ctx)
        if err:
            return err
        try:
            messages = service.get_messages(channel, limit=limit)
            if not messages:
                return f"No messages in {channel}. Run 'homie start' to sync first."
            lines = [f"**Recent messages in {channel}:**"]
            for m in messages:
                lines.append(f"  [{m.get('sender', '?')}] {m.get('text', '')[:120]}")
            return "\n".join(lines)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not get messages: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="social",
        description="Social messaging (channels, recent)",
        args_spec="channels|recent",
        subcommands={
            "channels": SlashCommand(name="channels", description="List channels", handler_fn=_handle_social_channels),
            "recent": SlashCommand(name="recent", description="Recent messages from a channel", args_spec="<channel_id> [--limit N]", handler_fn=_handle_social_recent),
        },
    ))
