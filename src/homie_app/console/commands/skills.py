"""Handler for /skills slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_skills(args: str, **ctx) -> str:
    try:
        from homie_core.skills.loader import SkillLoader
        loader = SkillLoader()
        skills = loader.scan()
        if not skills:
            return "No skills installed. Drop SKILL.md files into ~/.homie/skills/"
        return loader.build_skills_index()
    except Exception as e:
        return f"Could not load skills: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(name="skills", description="List installed skills", handler_fn=_handle_skills))
