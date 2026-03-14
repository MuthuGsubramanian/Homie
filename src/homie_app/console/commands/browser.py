"""Handler for /browser slash command — browser history."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _get_browser_service(ctx):
    """Build BrowserHistoryService from context config."""
    from pathlib import Path
    from homie_core.vault.secure_vault import SecureVault
    from homie_core.browser import BrowserHistoryService
    cfg = ctx.get("config")
    if not cfg:
        return None, None, "No configuration loaded."
    storage = Path(cfg.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()
    service = BrowserHistoryService(vault=vault)
    service.initialize()
    return service, vault, None


def _handle_browser_enable(args: str, **ctx) -> str:
    browsers_str = args.strip() or "chrome"
    browsers = [b.strip() for b in browsers_str.replace("--browsers", "").strip().split(",")]
    try:
        service, vault, err = _get_browser_service(ctx)
        if err:
            return err
        try:
            result = service.configure(enabled=True, browsers=browsers)
            return f"Browser history enabled for: {', '.join(result['browsers'])}"
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not enable browser history: {e}"


def _handle_browser_disable(args: str, **ctx) -> str:
    try:
        service, vault, err = _get_browser_service(ctx)
        if err:
            return err
        try:
            service.configure(enabled=False)
            return "Browser history disabled."
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not disable browser history: {e}"


def _handle_browser_config(args: str, **ctx) -> str:
    import json
    parts = args.strip().split()
    exclude = ""
    retention = 0
    if "--exclude" in parts:
        exclude = parts[parts.index("--exclude") + 1]
    if "--retention" in parts:
        retention = int(parts[parts.index("--retention") + 1])
    try:
        service, vault, err = _get_browser_service(ctx)
        if err:
            return err
        try:
            kwargs = {}
            if exclude:
                kwargs["exclude_domains"] = [d.strip() for d in exclude.split(",")]
            if retention:
                kwargs["retention_days"] = retention
            if kwargs:
                result = service.configure(**kwargs)
            else:
                result = service.get_config()
            return json.dumps(result, indent=2)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not configure browser history: {e}"


def _handle_browser_history(args: str, **ctx) -> str:
    parts = args.strip().split()
    limit = 50
    domain = ""
    if "--limit" in parts:
        limit = int(parts[parts.index("--limit") + 1])
    if "--domain" in parts:
        domain = parts[parts.index("--domain") + 1]
    try:
        service, vault, err = _get_browser_service(ctx)
        if err:
            return err
        try:
            entries = service.get_history(limit=limit, domain=domain or None)
            if not entries:
                return "No history entries found."
            lines = ["**Browser History:**"]
            for e in entries:
                lines.append(f"  [{e.get('browser')}] {e.get('title', '')[:50]} — {e.get('url', '')[:80]}")
            return "\n".join(lines)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not get browser history: {e}"


def _handle_browser_scan(args: str, **ctx) -> str:
    import json
    try:
        service, vault, err = _get_browser_service(ctx)
        if err:
            return err
        try:
            result = service.scan()
            return json.dumps(result, indent=2)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not scan browser history: {e}"


def _handle_browser_patterns(args: str, **ctx) -> str:
    import json
    try:
        service, vault, err = _get_browser_service(ctx)
        if err:
            return err
        try:
            result = service.get_patterns()
            return json.dumps(result, indent=2)
        finally:
            vault.lock()
    except Exception as e:
        return f"Could not get browsing patterns: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="browser",
        description="Browser history (enable, disable, config, history, scan, patterns)",
        args_spec="enable|disable|config|history|scan|patterns",
        subcommands={
            "enable": SlashCommand(name="enable", description="Enable browser history tracking", args_spec="[--browsers chrome,firefox]", handler_fn=_handle_browser_enable),
            "disable": SlashCommand(name="disable", description="Disable browser history tracking", handler_fn=_handle_browser_disable),
            "config": SlashCommand(name="config", description="Configure browser history", args_spec="[--exclude <domains>] [--retention <days>]", handler_fn=_handle_browser_config),
            "history": SlashCommand(name="history", description="View browsing history", args_spec="[--limit N] [--domain <domain>]", handler_fn=_handle_browser_history),
            "scan": SlashCommand(name="scan", description="Full history scan", handler_fn=_handle_browser_scan),
            "patterns": SlashCommand(name="patterns", description="Browsing patterns analysis", handler_fn=_handle_browser_patterns),
        },
    ))
