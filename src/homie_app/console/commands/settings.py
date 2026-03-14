"""Handler for /settings slash command — interactive settings menu."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


_CATEGORIES = [
    "LLM & Model",
    "Voice",
    "User Profile",
    "Screen Reader",
    "Email & Socials",
    "Privacy",
    "Plugins",
    "Notifications",
    "Service Mode",
    "Location",
    "Back",
]


def _ask_choice(title: str, options: list[str]) -> int:
    print(f"\n  {title}")
    print("  " + "-" * len(title))
    for i, opt in enumerate(options):
        print(f"    {i}: {opt}")
    while True:
        try:
            raw = input(f"  Choose [0-{len(options)-1}]: ").strip()
            choice = int(raw)
            if 0 <= choice < len(options):
                return choice
        except (ValueError, EOFError, KeyboardInterrupt):
            return len(options) - 1  # Back


def _handle_settings(args: str, **ctx) -> str:
    cfg = ctx.get("config")
    if not cfg:
        return "No configuration loaded."

    if args.strip():
        cat = args.strip().lower()
        for i, name in enumerate(_CATEGORIES[:-1]):
            if cat in name.lower():
                _run_step(i, cfg)
                _save(cfg, ctx.get("config_path"))
                return f"Settings updated: {name}"
        return f"Unknown category: {args}. Available: {', '.join(_CATEGORIES[:-1])}"

    while True:
        choice = _ask_choice("Homie Settings", _CATEGORIES)
        if choice == len(_CATEGORIES) - 1:
            break
        _run_step(choice, cfg)
        _save(cfg, ctx.get("config_path"))

    return ""


def _run_step(choice: int, cfg) -> None:
    try:
        from homie_app.init import (
            _step_user_profile, _step_screen_reader,
            _step_email, _step_social_connections,
            _step_privacy, _step_plugins,
            _step_service_mode,
        )

        steps = {
            2: _step_user_profile,
            3: _step_screen_reader,
            4: lambda c: (_step_email(c), _step_social_connections(c)),
            5: _step_privacy,
            6: _step_plugins,
            8: _step_service_mode,
        }

        fn = steps.get(choice)
        if fn:
            fn(cfg)
        elif choice == 0:
            print("  LLM & Model settings — use /model to manage models.")
        elif choice == 1:
            print("  Voice settings — use /voice to manage voice pipeline.")
        elif choice == 7:
            print("  Notification settings — not yet implemented.")
        elif choice == 9:
            print("  Location settings — use /location set <city>.")
    except Exception as e:
        print(f"  Error in settings: {e}")


def _save(cfg, config_path: str | None) -> None:
    try:
        from homie_app.init import _save_config
        path = config_path or "homie.config.yaml"
        _save_config(cfg, path)
    except Exception:
        pass


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="settings",
        description="Configure Homie (LLM, voice, privacy, email, location, etc.)",
        args_spec="[category]",
        handler_fn=_handle_settings,
    ))
