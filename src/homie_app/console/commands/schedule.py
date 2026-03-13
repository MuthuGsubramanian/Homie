"""Handler for /schedule slash command."""
from __future__ import annotations
from homie_app.console.router import SlashCommand, SlashCommandRouter


def _handle_schedule_add(args: str, **ctx) -> str:
    parts = args.strip().split(maxsplit=2)
    if len(parts) < 3:
        return "Usage: /schedule add <name> <schedule> <prompt>\n  Example: /schedule add remind_break every_2h Take a break"
    try:
        from homie_core.scheduler.cron import JobStore
        job_store = JobStore()
        job = job_store.create_job(name=parts[0], prompt=parts[2], schedule=parts[1])
        return f"Scheduled: '{job.name}' ({parts[1]}) — next run: {job.next_run}"
    except Exception as e:
        return f"Could not schedule: {e}"


def _handle_schedule_list(args: str, **ctx) -> str:
    try:
        from homie_core.scheduler.cron import JobStore
        job_store = JobStore()
        jobs = job_store.list_jobs()
        if not jobs:
            return "No scheduled jobs."
        lines = ["**Scheduled Jobs:**"]
        for j in jobs:
            lines.append(f"  [{j.id}] {j.name} — {j.schedule} — next: {j.next_run}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not list jobs: {e}"


def _handle_schedule_remove(args: str, **ctx) -> str:
    job_id = args.strip()
    if not job_id:
        return "Usage: /schedule remove <job_id>"
    try:
        from homie_core.scheduler.cron import JobStore
        job_store = JobStore()
        job_store.remove_job(job_id)
        return f"Removed job: {job_id}"
    except Exception as e:
        return f"Could not remove job: {e}"


def register(router: SlashCommandRouter, ctx: dict) -> None:
    router.register(SlashCommand(
        name="schedule",
        description="Manage scheduled tasks",
        args_spec="add|list|remove",
        subcommands={
            "add": SlashCommand(name="add", description="Add a scheduled job", args_spec="<name> <schedule> <prompt>", handler_fn=_handle_schedule_add),
            "list": SlashCommand(name="list", description="List scheduled jobs", handler_fn=_handle_schedule_list),
            "remove": SlashCommand(name="remove", description="Remove a scheduled job", args_spec="<job_id>", handler_fn=_handle_schedule_remove),
        },
    ))
