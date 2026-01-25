from __future__ import annotations

from typing import Dict

from homie.config import HomieConfig, cfg_get
from homie.controller.orchestrator import Orchestrator

BUILTIN_TASKS: Dict[str, Dict[str, str]] = {
    "linux_cleanup_cache": {
        "command": "sudo apt-get clean && sudo journalctl --vacuum-time=7d && sudo find /tmp -type f -mtime +7 -delete",
        "reason": "automation:linux_cleanup_cache",
    },
    "linux_check_status": {
        "command": "uptime && df -h / && free -h && (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true)",
        "reason": "automation:linux_check_status",
    },
    "windows_cleanup_windows": {
        "command": 'powershell.exe -Command "Clear-RecycleBin -Force; ipconfig /flushdns"',
        "reason": "automation:windows_cleanup_windows",
    },
}


def _dispatch(orchestrator: Orchestrator, target: str, command: str, reason: str) -> None:
    orchestrator.dispatch(
        {
            "action": "run_command",
            "target": target,
            "command": command,
            "reason": reason,
            "risk_class": "low",
            "autonomy_level": "suggest",
        },
        dry_run=False,
    )


def register_automation_jobs(scheduler, orchestrator: Orchestrator, cfg: HomieConfig) -> None:
    if not cfg_get(cfg, "automations", "enabled", default=False):
        return
    schedules = cfg_get(cfg, "automations", "schedules", default=[]) or []
    for job in schedules:
        name = job.get("name")
        cron = job.get("cron")
        target = job.get("target")
        task = job.get("task") or name
        if not name or not cron or not target or task not in BUILTIN_TASKS:
            continue
        details = BUILTIN_TASKS[task]
        scheduler.add_cron_job(
            name=name,
            cron=cron,
            func=_dispatch,
            args=[orchestrator, target, details["command"], details["reason"]],
            kwargs={},
        )


__all__ = ["register_automation_jobs", "BUILTIN_TASKS"]
