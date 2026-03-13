"""Handler for /insights slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_insights(args: str, **ctx) -> str:
    try:
        from pathlib import Path
        from homie_core.analytics.insights import InsightsEngine
        cfg = ctx.get("config")
        days = 30
        if args.strip():
            try:
                days = int(args.strip().replace("--days", "").strip())
            except ValueError:
                pass
        engine = InsightsEngine(Path(cfg.storage.path).expanduser())
        insights = engine.generate_insights(days=days)
        return engine.format_terminal(insights)
    except Exception as e:
        return f"Could not generate insights: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(name="insights", description="Show usage analytics and stats", args_spec="[--days N]", handler_fn=_handle_insights))
