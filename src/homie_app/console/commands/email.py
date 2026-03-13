"""Handler for /email slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_email_summary(args: str, **ctx) -> str:
    try:
        from homie_core.email import EmailService
        from homie_core.vault.secure_vault import SecureVault
        from pathlib import Path
        import sqlite3

        cfg = ctx.get("config")
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        storage_path = Path(cfg.storage.path).expanduser()
        cache_conn = sqlite3.connect(str(storage_path / "cache.db"))

        service = EmailService(vault, cache_conn)
        accounts = service.initialize()
        if not accounts:
            return "No email accounts connected. Use /connect gmail first."

        days = 1
        if args.strip():
            try:
                days = int(args.strip().split("--days")[-1].strip() if "--days" in args else args.strip())
            except ValueError:
                days = 1

        summary = service.get_summary(days=days)
        lines = [f"**Email Summary ({days} day{'s' if days > 1 else ''}):**"]
        lines.append(f"  Total: {summary.get('total', 0)}")
        lines.append(f"  Unread: {summary.get('unread', 0)}")
        hp = summary.get("high_priority", [])
        if hp:
            lines.append(f"  High priority: {len(hp)}")
            for msg in hp[:5]:
                lines.append(f"    - {msg.get('subject', '(no subject)')}")
        return "\n".join(lines)
    except Exception as e:
        return f"Email summary failed: {e}"


def _handle_email_sync(args: str, **ctx) -> str:
    try:
        from homie_core.email import EmailService
        from homie_core.vault.secure_vault import SecureVault
        from pathlib import Path
        import sqlite3

        cfg = ctx.get("config")
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        storage_path = Path(cfg.storage.path).expanduser()
        cache_conn = sqlite3.connect(str(storage_path / "cache.db"))

        service = EmailService(vault, cache_conn)
        accounts = service.initialize()
        if not accounts:
            return "No email accounts connected. Use /connect gmail first."

        result = service.sync_tick()
        return f"Sync complete: {result}"
    except Exception as e:
        return f"Email sync failed: {e}"


def _handle_email_config(args: str, **ctx) -> str:
    try:
        from homie_core.vault.secure_vault import SecureVault
        vault = ctx.get("vault")
        if not vault:
            vault = SecureVault()
            vault.unlock()

        creds = vault.list_credentials("gmail")
        if not creds:
            return "No Gmail accounts configured. Use /connect gmail."
        lines = ["**Email Configuration:**"]
        for c in creds:
            lines.append(f"  Account: {c.account_id}")
            lines.append(f"  Active: {c.active}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not read email config: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="email",
        description="Email operations (summary, sync, config)",
        args_spec="summary|sync|config",
        subcommands={
            "summary": SlashCommand(name="summary", description="Email summary", args_spec="[--days N]", handler_fn=_handle_email_summary),
            "sync": SlashCommand(name="sync", description="Force sync now", handler_fn=_handle_email_sync),
            "config": SlashCommand(name="config", description="Show email settings", handler_fn=_handle_email_config),
        },
    ))
