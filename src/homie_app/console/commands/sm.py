"""Handler for /sm slash command — social media profile operations."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _get_sm_service(ctx):
    """Build SocialMediaService from context config."""
    from pathlib import Path
    from homie_core.vault.secure_vault import SecureVault
    from homie_core.social_media import SocialMediaService
    cfg = ctx.get("config")
    if not cfg:
        return None, None, "No configuration loaded."
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()
    service = SocialMediaService(vault=vault)
    service.initialize()
    return service, vault, None


def _handle_sm_feed(args: str, **ctx) -> str:
    parts = args.strip().split()
    platform = "all"
    limit = 20
    if "--platform" in parts:
        platform = parts[parts.index("--platform") + 1]
    if "--limit" in parts:
        limit = int(parts[parts.index("--limit") + 1])
    try:
        service, vault, err = _get_sm_service(ctx)
        if err:
            return err
        try:
            results = service.get_feed(platform=platform, limit=limit)
            if not results:
                return "No posts found."
            lines = ["**Social Media Feed:**"]
            for post in results:
                lines.append(f"  [{post.get('platform')}] {post.get('author')}: {post.get('content', '')[:100]}")
            return "\n".join(lines)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not get feed: {e}"


def _handle_sm_profile(args: str, **ctx) -> str:
    import json
    parts = args.strip().split()
    if not parts or "--platform" not in parts:
        return "Usage: /sm profile --platform <platform> [--username <username>]"
    platform = parts[parts.index("--platform") + 1]
    username = ""
    if "--username" in parts:
        username = parts[parts.index("--username") + 1]
    try:
        service, vault, err = _get_sm_service(ctx)
        if err:
            return err
        try:
            info = service.get_profile(platform, username=username or None)
            return json.dumps(info, indent=2)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not get profile: {e}"


def _handle_sm_scan(args: str, **ctx) -> str:
    import json
    try:
        service, vault, err = _get_sm_service(ctx)
        if err:
            return err
        try:
            result = service.scan_profiles()
            return json.dumps(result, indent=2)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not scan profiles: {e}"


def _handle_sm_publish(args: str, **ctx) -> str:
    import json
    parts = args.strip().split(maxsplit=1)
    if len(parts) < 2:
        return "Usage: /sm publish <platform> <content>"
    platform = parts[0]
    content = parts[1]
    try:
        service, vault, err = _get_sm_service(ctx)
        if err:
            return err
        try:
            result = service.publish(platform, content)
            return json.dumps(result, indent=2)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not publish: {e}"


def _handle_sm_dms(args: str, **ctx) -> str:
    parts = args.strip().split()
    if not parts:
        return "Usage: /sm dms <platform> [--conversation <id>]"
    platform = parts[0]
    conversation = ""
    if "--conversation" in parts:
        conversation = parts[parts.index("--conversation") + 1]
    try:
        service, vault, err = _get_sm_service(ctx)
        if err:
            return err
        try:
            if conversation:
                msgs = service.get_dms(platform, conversation)
                lines = [f"**DMs in conversation {conversation}:**"]
                for m in msgs:
                    lines.append(f"  [{m.get('sender')}]: {m.get('content', '')[:100]}")
                return "\n".join(lines)
            else:
                convos = service.get_conversations(platform)
                lines = [f"**DM Conversations on {platform}:**"]
                for c in convos:
                    lines.append(f"  [{c.get('id')}] {', '.join(c.get('participants', []))}: "
                                 f"{c.get('last_message_preview', '')[:50]}")
                return "\n".join(lines)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not get DMs: {e}"


def _handle_sm_send_dm(args: str, **ctx) -> str:
    import json
    parts = args.strip().split(maxsplit=2)
    if len(parts) < 3:
        return "Usage: /sm send-dm <platform> <recipient> <text>"
    platform, recipient, text = parts[0], parts[1], parts[2]
    try:
        service, vault, err = _get_sm_service(ctx)
        if err:
            return err
        try:
            result = service.send_dm(platform, recipient, text)
            return json.dumps(result, indent=2)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not send DM: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="sm",
        description="Social media operations (feed, profile, scan, publish, dms, send-dm)",
        args_spec="feed|profile|scan|publish|dms|send-dm",
        subcommands={
            "feed": SlashCommand(name="feed", description="Get social media feed", args_spec="[--platform <platform>] [--limit N]", handler_fn=_handle_sm_feed),
            "profile": SlashCommand(name="profile", description="Get profile info", args_spec="--platform <platform> [--username <user>]", handler_fn=_handle_sm_profile),
            "scan": SlashCommand(name="scan", description="Full profile scan + intelligence", handler_fn=_handle_sm_scan),
            "publish": SlashCommand(name="publish", description="Publish a post", args_spec="<platform> <content>", handler_fn=_handle_sm_publish),
            "dms": SlashCommand(name="dms", description="View DM conversations", args_spec="<platform> [--conversation <id>]", handler_fn=_handle_sm_dms),
            "send-dm": SlashCommand(name="send-dm", description="Send a DM", args_spec="<platform> <recipient> <text>", handler_fn=_handle_sm_send_dm),
        },
    ))
