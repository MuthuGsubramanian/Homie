"""Handler for /voice slash command — voice interaction mode."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_voice_status(args: str, **ctx) -> str:
    try:
        cfg = ctx.get("config")
        if not cfg:
            return "No configuration loaded."
        from homie_core.voice.voice_manager import VoiceManager
        mgr = VoiceManager(config=cfg.voice, on_query=lambda t: iter(["Voice status check"]))
        return mgr.status_report()
    except Exception as e:
        return f"Could not get voice status: {e}"


def _handle_voice_enable(args: str, **ctx) -> str:
    try:
        cfg = ctx.get("config")
        if cfg:
            cfg.voice.enabled = True
            # Apply optional mode/tts from args
            parts = args.strip().split()
            if "--mode" in parts:
                cfg.voice.mode = parts[parts.index("--mode") + 1]
            if "--tts" in parts:
                cfg.voice.tts_mode = parts[parts.index("--tts") + 1]
            if "--lang" in parts:
                cfg.voice.stt_language = parts[parts.index("--lang") + 1]
        return "Voice enabled. Update homie.config.yaml to persist."
    except Exception as e:
        return f"Could not enable voice: {e}"


def _handle_voice_disable(args: str, **ctx) -> str:
    try:
        cfg = ctx.get("config")
        if cfg:
            cfg.voice.enabled = False
        return "Voice disabled. Update homie.config.yaml to persist."
    except Exception as e:
        return f"Could not disable voice: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="voice",
        description="Voice interaction mode (status, enable, disable)",
        args_spec="status|enable|disable",
        subcommands={
            "status": SlashCommand(name="status", description="Show voice component status", handler_fn=_handle_voice_status),
            "enable": SlashCommand(name="enable", description="Enable voice", args_spec="[--mode hybrid|wake_word|push_to_talk] [--tts auto|fast|quality] [--lang <lang>]", handler_fn=_handle_voice_enable),
            "disable": SlashCommand(name="disable", description="Disable voice", handler_fn=_handle_voice_disable),
        },
    ))
