from __future__ import annotations

from typing import Optional

from homie_core.intelligence.session_tracker import SessionTracker
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.utils import utc_now


def _greeting(hour: int) -> str:
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


class BriefingGenerator:
    """Generates morning briefings and end-of-day digests."""

    def __init__(self, session_tracker: SessionTracker,
                 user_name: str = ""):
        self._tracker = session_tracker
        self._user_name = user_name

    def morning_briefing(self) -> str:
        now = utc_now()
        name_part = f", {self._user_name}" if self._user_name else ""
        greeting = f"{_greeting(now.hour)}{name_part}!"

        lines = [greeting, ""]

        resumption = self._tracker.get_resumption_summary()
        if resumption:
            lines.append(resumption)
        else:
            lines.append("No previous session found. Starting fresh!")

        lines.append("")
        lines.append(f"Today is {now.strftime('%A, %B %d')}. Ready when you are.")

        return "\n".join(lines)

    def end_of_day_digest(self, task_graph: TaskGraph,
                          apps_used: dict[str, float] | None = None,
                          switch_count: int = 0) -> str:
        digest = self._tracker.generate_digest(
            task_graph, apps_used=apps_used, switch_count=switch_count,
        )

        # Save session for tomorrow's morning briefing
        self._tracker.save_session(task_graph, apps_used=apps_used)

        name_part = f", {self._user_name}" if self._user_name else ""
        return f"Wrapping up{name_part}.\n\n{digest}"
