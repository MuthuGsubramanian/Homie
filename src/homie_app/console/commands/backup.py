"""Handler for /backup and /restore slash commands."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_backup(args: str, **ctx) -> str:
    backup_path = args.strip()
    if not backup_path:
        return "Usage: /backup <path>\n  Example: /backup ~/homie-backup.tar.gz"
    try:
        from pathlib import Path
        import getpass
        from homie_core.storage.backup import BackupManager
        cfg = ctx.get("config")
        if not cfg:
            return "No configuration loaded."
        passphrase = getpass.getpass("Backup passphrase: ")
        mgr = BackupManager()
        out = mgr.create_backup(Path(cfg.storage.path), Path(backup_path), passphrase)
        return f"Backup created at {out}"
    except Exception as e:
        return f"Could not create backup: {e}"


def _handle_restore(args: str, **ctx) -> str:
    restore_path = args.strip()
    if not restore_path:
        return "Usage: /restore <path>\n  Example: /restore ~/homie-backup.tar.gz"
    try:
        from pathlib import Path
        import getpass
        from homie_core.storage.backup import BackupManager
        cfg = ctx.get("config")
        if not cfg:
            return "No configuration loaded."
        passphrase = getpass.getpass("Backup passphrase: ")
        mgr = BackupManager()
        mgr.restore_backup(Path(restore_path), Path(cfg.storage.path), passphrase)
        return f"Restored to {cfg.storage.path}"
    except Exception as e:
        return f"Could not restore backup: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="backup",
        description="Create encrypted backup",
        args_spec="<path>",
        handler_fn=_handle_backup,
    ))
    router.register(SlashCommand(
        name="restore",
        description="Restore from backup",
        args_spec="<path>",
        handler_fn=_handle_restore,
    ))
