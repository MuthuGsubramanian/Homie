"""Usage analytics and insights for the Homie AI assistant.

Aggregates data from session files, episodic memory, and semantic memory
to produce usage statistics and formatted reports.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# Common words to exclude when extracting topics from summaries.
_STOP_WORDS = frozenset(
    "the a an and or but in on at to for of is was were be been being "
    "it its i me my you your he she they we our this that with from by "
    "as not no do did does will would can could have has had are am "
    "about so if then than also just very too all any each some more "
    "what when where how which who whom whose there here".split()
)


@dataclass
class SessionInsights:
    """Aggregated usage insights over a time window."""

    total_sessions: int = 0
    total_messages: int = 0
    total_facts_learned: int = 0
    total_episodes: int = 0
    avg_session_length_turns: float = 0.0
    top_topics: list[tuple[str, int]] = field(default_factory=list)
    top_tools_used: list[tuple[str, int]] = field(default_factory=list)
    activity_breakdown: dict[str, float] = field(default_factory=dict)
    hourly_activity: dict[int, int] = field(default_factory=dict)
    daily_streak: int = 0
    most_active_hour: int = 0
    mood_distribution: dict[str, int] = field(default_factory=dict)
    facts_by_tag: dict[str, int] = field(default_factory=dict)


def _ascii_bar(label: str, value: int, max_value: int, width: int = 30) -> str:
    """Render a single ASCII bar-chart row.

    Args:
        label: Row label displayed left-aligned.
        value: Numeric value for this row.
        max_value: The maximum value across all rows (used for scaling).
        width: Character width of the longest bar.

    Returns:
        A formatted string like ``"  label  |######              | 42"``
    """
    if max_value <= 0:
        filled = 0
    else:
        filled = round(value / max_value * width)
    bar = "#" * filled + " " * (width - filled)
    return f"  {label:<8s} |{bar}| {value}"


class InsightsEngine:
    """Analyses Homie's stored data and produces usage insights.

    The engine reads from two data sources:

    1. **Session files** – JSON files written by
       :class:`homie_core.intelligence.session_tracker.SessionTracker` into a
       ``sessions/`` sub-directory under *storage_dir*.
    2. **SQLite database** – the ``homie.db`` file managed by
       :class:`homie_core.storage.database.Database`, containing episodic
       memory (``episodes_meta``), semantic memory (``semantic_memory``), and
       other tables.

    Both sources are optional; the engine returns partial results when data is
    missing.
    """

    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = Path(storage_dir)
        self._sessions_dir = self._storage_dir / "sessions"
        self._db_path = self._storage_dir / "homie.db"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_insights(self, days: int = 30) -> SessionInsights:
        """Analyse the last *days* days of data and return insights.

        Args:
            days: Number of past days to include (default 30).

        Returns:
            A :class:`SessionInsights` dataclass with all computed fields.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        insights = SessionInsights()

        sessions = self._load_sessions(cutoff)
        episodes = self._load_episodes(cutoff)
        facts = self._load_facts(cutoff)

        # --- Session-level metrics ---
        insights.total_sessions = len(sessions)

        turn_counts: list[int] = []
        topic_counter: Counter[str] = Counter()
        hourly: Counter[int] = Counter()
        session_dates: set[str] = set()
        activity_counter: Counter[str] = Counter()

        for sess in sessions:
            turns = sess.get("turn_count", 0)
            if turns == 0:
                # Fallback: count messages list if present.
                turns = len(sess.get("messages", []))
            turn_counts.append(turns)
            insights.total_messages += turns

            # Session timestamp -> hourly activity & daily streak
            ts = self._parse_timestamp(sess.get("saved_at", ""))
            if ts:
                hourly[ts.hour] += 1
                session_dates.add(ts.strftime("%Y-%m-%d"))

            # Topics from task graph summaries / apps_used keys
            summary_text = self._extract_summary_text(sess)
            topic_counter.update(self._tokenize_topics(summary_text))

            # Activity breakdown from apps_used
            apps_used = sess.get("apps_used", {})
            for app, secs in apps_used.items():
                activity_counter[app] += secs

        if turn_counts:
            insights.avg_session_length_turns = sum(turn_counts) / len(turn_counts)

        # --- Episode-level metrics ---
        insights.total_episodes = len(episodes)
        tool_counter: Counter[str] = Counter()
        mood_counter: Counter[str] = Counter()

        for ep in episodes:
            tags = ep.get("context_tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except (json.JSONDecodeError, TypeError):
                    tags = []
            tool_counter.update(tags)

            mood = ep.get("mood")
            if mood:
                mood_counter[mood] += 1

            # Also extract topics from episode summaries
            summary = ep.get("summary", "")
            topic_counter.update(self._tokenize_topics(summary))

            # Hourly activity from episode timestamps
            ts = self._parse_timestamp(ep.get("created_at", ""))
            if ts:
                hourly[ts.hour] += 1
                session_dates.add(ts.strftime("%Y-%m-%d"))

        # --- Fact-level metrics ---
        insights.total_facts_learned = len(facts)
        tag_counter: Counter[str] = Counter()
        for fact in facts:
            tags = fact.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except (json.JSONDecodeError, TypeError):
                    tags = []
            tag_counter.update(tags)

        # --- Assemble final insights ---
        insights.top_topics = topic_counter.most_common(10)
        insights.top_tools_used = tool_counter.most_common(10)
        insights.mood_distribution = dict(mood_counter)
        insights.facts_by_tag = dict(tag_counter)
        insights.hourly_activity = dict(sorted(hourly.items()))
        insights.most_active_hour = hourly.most_common(1)[0][0] if hourly else 0
        insights.daily_streak = self._compute_streak(session_dates)

        # Activity breakdown as percentages
        total_secs = sum(activity_counter.values())
        if total_secs > 0:
            insights.activity_breakdown = {
                app: round(secs / total_secs * 100, 1)
                for app, secs in activity_counter.most_common()
            }

        return insights

    def format_terminal(self, insights: SessionInsights) -> str:
        """Format insights as a rich terminal report with ASCII bar charts.

        Args:
            insights: The insights dataclass to render.

        Returns:
            A multi-line string suitable for printing to a terminal.
        """
        lines: list[str] = []
        sep = "-" * 60

        lines.append(sep)
        lines.append("  HOMIE USAGE INSIGHTS")
        lines.append(sep)

        # Overview
        lines.append("")
        lines.append("  Overview")
        lines.append(f"    Sessions:       {insights.total_sessions}")
        lines.append(f"    Messages:       {insights.total_messages}")
        lines.append(f"    Episodes:       {insights.total_episodes}")
        lines.append(f"    Facts learned:  {insights.total_facts_learned}")
        lines.append(f"    Avg turns/sess: {insights.avg_session_length_turns:.1f}")
        lines.append(f"    Daily streak:   {insights.daily_streak} day(s)")
        lines.append(f"    Most active hr: {insights.most_active_hour:02d}:00")

        # Top topics
        if insights.top_topics:
            lines.append("")
            lines.append("  Top Topics")
            for topic, count in insights.top_topics[:8]:
                lines.append(f"    {topic:<20s} {count}")

        # Top tools used
        if insights.top_tools_used:
            lines.append("")
            lines.append("  Top Tools Used")
            for tool, count in insights.top_tools_used[:8]:
                lines.append(f"    {tool:<20s} {count}")

        # Activity breakdown
        if insights.activity_breakdown:
            lines.append("")
            lines.append("  Activity Breakdown")
            for app, pct in list(insights.activity_breakdown.items())[:8]:
                lines.append(f"    {app:<20s} {pct:5.1f}%")

        # Hourly activity bar chart
        if insights.hourly_activity:
            lines.append("")
            lines.append("  Hourly Activity")
            max_val = max(insights.hourly_activity.values()) if insights.hourly_activity else 1
            for hour in range(24):
                count = insights.hourly_activity.get(hour, 0)
                label = f"{hour:02d}:00"
                lines.append(_ascii_bar(label, count, max_val))

        # Mood distribution bar chart
        if insights.mood_distribution:
            lines.append("")
            lines.append("  Mood Distribution")
            max_val = max(insights.mood_distribution.values()) if insights.mood_distribution else 1
            for mood, count in sorted(insights.mood_distribution.items(), key=lambda x: -x[1]):
                lines.append(_ascii_bar(mood, count, max_val))

        # Facts by tag
        if insights.facts_by_tag:
            lines.append("")
            lines.append("  Facts by Tag")
            for tag, count in sorted(insights.facts_by_tag.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"    {tag:<20s} {count}")

        lines.append("")
        lines.append(sep)
        return "\n".join(lines)

    def format_compact(self, insights: SessionInsights) -> str:
        """Format insights as a single-paragraph summary.

        Args:
            insights: The insights dataclass to render.

        Returns:
            A concise one-paragraph string.
        """
        parts: list[str] = []

        parts.append(
            f"Over the analysis window you had {insights.total_sessions} session(s) "
            f"with {insights.total_messages} message(s) "
            f"(avg {insights.avg_session_length_turns:.1f} turns per session)."
        )

        if insights.daily_streak > 0:
            parts.append(f"Your current daily streak is {insights.daily_streak} day(s).")

        parts.append(
            f"Homie recorded {insights.total_episodes} episode(s) "
            f"and learned {insights.total_facts_learned} fact(s)."
        )

        if insights.top_topics:
            top = ", ".join(t for t, _ in insights.top_topics[:3])
            parts.append(f"Top topics: {top}.")

        if insights.most_active_hour:
            parts.append(f"You're most active around {insights.most_active_hour:02d}:00.")

        if insights.mood_distribution:
            dominant = max(insights.mood_distribution, key=insights.mood_distribution.get)
            parts.append(f"Most common mood: {dominant}.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_sessions(self, cutoff: datetime) -> list[dict[str, Any]]:
        """Load session JSON files saved after *cutoff*.

        Reads both individual session files from ``sessions/`` and the
        ``last_session.json`` written by SessionTracker.
        """
        sessions: list[dict[str, Any]] = []

        # Individual session files in sessions/ directory
        if self._sessions_dir.is_dir():
            for path in self._sessions_dir.glob("*.json"):
                data = self._read_json(path)
                if data is None:
                    continue
                ts = self._parse_timestamp(data.get("saved_at", ""))
                if ts and ts >= cutoff:
                    sessions.append(data)

        # Also check the last_session.json written by SessionTracker
        last_session = self._storage_dir / "last_session.json"
        if last_session.is_file():
            data = self._read_json(last_session)
            if data is not None:
                ts = self._parse_timestamp(data.get("saved_at", ""))
                if ts and ts >= cutoff:
                    sessions.append(data)

        return sessions

    def _load_episodes(self, cutoff: datetime) -> list[dict[str, Any]]:
        """Load episode records from the SQLite database."""
        if not self._db_path.is_file():
            return []
        cutoff_str = cutoff.isoformat()
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM episodes_meta WHERE created_at >= ? ORDER BY created_at",
                (cutoff_str,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return []

    def _load_facts(self, cutoff: datetime) -> list[dict[str, Any]]:
        """Load semantic memory facts from the SQLite database."""
        if not self._db_path.is_file():
            return []
        cutoff_str = cutoff.isoformat()
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM semantic_memory WHERE created_at >= ? ORDER BY created_at",
                (cutoff_str,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return []

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        """Safely read and parse a JSON file, returning *None* on failure."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _parse_timestamp(ts_str: str) -> datetime | None:
        """Parse an ISO-8601 timestamp string into a timezone-aware datetime."""
        if not ts_str:
            return None
        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_summary_text(session: dict[str, Any]) -> str:
        """Pull human-readable text from a session dict for topic extraction."""
        parts: list[str] = []

        # Task graph may contain task descriptions
        tg = session.get("task_graph", {})
        if isinstance(tg, dict):
            for task in tg.get("tasks", []):
                if isinstance(task, dict):
                    parts.append(task.get("description", ""))
                    parts.extend(task.get("apps", []))

        # Apps used keys can serve as topic signals
        for app in session.get("apps_used", {}):
            parts.append(app)

        return " ".join(parts)

    @staticmethod
    def _tokenize_topics(text: str) -> list[str]:
        """Extract meaningful tokens from *text* for topic counting.

        Filters out stop words and short tokens.
        """
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        return [t for t in tokens if t not in _STOP_WORDS]

    @staticmethod
    def _compute_streak(dates: set[str]) -> int:
        """Compute the current consecutive-day streak ending today or yesterday.

        Args:
            dates: A set of ``"YYYY-MM-DD"`` date strings.

        Returns:
            Number of consecutive days with at least one session.
        """
        if not dates:
            return 0

        today = datetime.now(timezone.utc).date()
        # Allow streak to include today or start from yesterday
        check = today
        if check.isoformat() not in dates:
            check = today - timedelta(days=1)
            if check.isoformat() not in dates:
                return 0

        streak = 0
        while check.isoformat() in dates:
            streak += 1
            check -= timedelta(days=1)

        return streak
