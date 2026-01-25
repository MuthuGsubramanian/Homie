from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


class AutomationScheduler:
    """Thin wrapper around APScheduler for HOMIE automations."""

    def __init__(self) -> None:
        self._sched = BackgroundScheduler()
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._sched.start(paused=False)
            self._started = True

    def stop(self) -> None:
        if self._started:
            self._sched.shutdown(wait=False)
            self._started = False

    def add_cron_job(
        self,
        name: str,
        cron: str,
        func: Callable,
        args: Optional[list] = None,
        kwargs: Optional[Dict] = None,
    ) -> None:
        trigger = CronTrigger.from_crontab(cron)
        self._sched.add_job(func, trigger, id=name, args=args or [], kwargs=kwargs or {}, replace_existing=True)
        logging.info("Scheduled job %s (%s)", name, cron)

    def remove(self, name: str) -> None:
        try:
            self._sched.remove_job(name)
        except Exception:  # noqa: BLE001
            logging.debug("Job %s not present", name)


__all__ = ["AutomationScheduler"]
