"""Tests for morning_briefing module."""
from __future__ import annotations

from unittest.mock import patch
from datetime import datetime

import pytest
from homie_core.intelligence.morning_briefing import (
    MorningBriefingOrchestrator,
    MorningBriefing,
    BriefingSection,
)


class TestShouldFire:
    def _orchestrator_at_hour(self, hour: int) -> MorningBriefingOrchestrator:
        orc = MorningBriefingOrchestrator(earliest_hour=6, latest_hour=10)
        return orc

    def test_returns_true_during_morning_window(self):
        orc = MorningBriefingOrchestrator(earliest_hour=6, latest_hour=10)
        with patch("homie_core.intelligence.morning_briefing.datetime") as mock_dt:
            mock_dt.now.side_effect = lambda: datetime(2026, 3, 21, 7, 30)
            assert orc.should_fire() is True

    def test_returns_false_before_morning_window(self):
        orc = MorningBriefingOrchestrator(earliest_hour=6, latest_hour=10)
        with patch("homie_core.intelligence.morning_briefing.datetime") as mock_dt:
            mock_dt.now.side_effect = lambda: datetime(2026, 3, 21, 5, 0)
            assert orc.should_fire() is False

    def test_returns_false_after_morning_window(self):
        orc = MorningBriefingOrchestrator(earliest_hour=6, latest_hour=10)
        with patch("homie_core.intelligence.morning_briefing.datetime") as mock_dt:
            mock_dt.now.side_effect = lambda: datetime(2026, 3, 21, 11, 0)
            assert orc.should_fire() is False

    def test_returns_false_after_already_fired_today(self):
        orc = MorningBriefingOrchestrator(earliest_hour=6, latest_hour=10)
        with patch("homie_core.intelligence.morning_briefing.datetime") as mock_dt:
            mock_dt.now.side_effect = lambda: datetime(2026, 3, 21, 7, 30)
            assert orc.should_fire() is True   # first call fires
            # generate sets _last_fired_date; simulate by calling should_fire again same day
            orc._last_fired_date = "2026-03-21"
            assert orc.should_fire() is False  # already fired today

    def test_fires_again_next_day(self):
        orc = MorningBriefingOrchestrator(earliest_hour=6, latest_hour=10)
        orc._last_fired_date = "2026-03-20"
        with patch("homie_core.intelligence.morning_briefing.datetime") as mock_dt:
            mock_dt.now.side_effect = lambda: datetime(2026, 3, 21, 8, 0)
            assert orc.should_fire() is True


class TestGenerate:
    def _make_orc(self) -> MorningBriefingOrchestrator:
        return MorningBriefingOrchestrator()

    def test_generate_with_commitments_includes_section(self):
        orc = self._make_orc()
        commitments = [{"text": "reply to Alice", "due_by": "today"}]
        briefing = orc.generate(commitments=commitments)
        titles = [s.title for s in briefing.sections]
        assert "Pending Commitments" in titles

    def test_generate_commitment_item_text(self):
        orc = self._make_orc()
        commitments = [{"text": "send invoice", "due_by": "Friday"}]
        briefing = orc.generate(commitments=commitments)
        section = next(s for s in briefing.sections if s.title == "Pending Commitments")
        assert any("send invoice" in item for item in section.items)
        assert any("Friday" in item for item in section.items)

    def test_generate_with_incomplete_tasks_includes_section(self):
        orc = self._make_orc()
        tasks = [{"description": "fix login bug"}]
        briefing = orc.generate(incomplete_tasks=tasks)
        titles = [s.title for s in briefing.sections]
        assert "Unfinished Tasks" in titles

    def test_generate_incomplete_tasks_fallback_key(self):
        orc = self._make_orc()
        tasks = [{"task": "deploy hotfix"}]
        briefing = orc.generate(incomplete_tasks=tasks)
        section = next(s for s in briefing.sections if s.title == "Unfinished Tasks")
        assert any("deploy hotfix" in item for item in section.items)

    def test_generate_with_session_summary_includes_section(self):
        orc = self._make_orc()
        briefing = orc.generate(session_summary="Worked on auth module all day.")
        titles = [s.title for s in briefing.sections]
        assert "Yesterday's Summary" in titles

    def test_generate_with_recent_facts_includes_section(self):
        orc = self._make_orc()
        facts = [{"fact": "Python 3.13 released"}]
        briefing = orc.generate(recent_facts=facts)
        titles = [s.title for s in briefing.sections]
        assert "Recently Learned" in titles

    def test_generate_recent_facts_capped_at_five(self):
        orc = self._make_orc()
        facts = [{"fact": f"fact {i}"} for i in range(10)]
        briefing = orc.generate(recent_facts=facts)
        section = next(s for s in briefing.sections if s.title == "Recently Learned")
        assert len(section.items) == 5

    def test_generate_with_no_data_returns_empty_sections(self):
        orc = self._make_orc()
        briefing = orc.generate()
        assert briefing.sections == []

    def test_generate_greeting_includes_user_name(self):
        orc = self._make_orc()
        briefing = orc.generate(user_name="Muthukumar")
        assert "Muthukumar" in briefing.greeting

    def test_generate_greeting_without_name(self):
        orc = self._make_orc()
        briefing = orc.generate()
        assert "Good morning" in briefing.greeting

    def test_generate_sets_last_fired_date(self):
        orc = self._make_orc()
        orc.generate()
        today = datetime.now().strftime("%Y-%m-%d")
        assert orc._last_fired_date == today

    def test_generate_sets_generated_at(self):
        orc = self._make_orc()
        briefing = orc.generate()
        assert briefing.generated_at != ""


class TestFormatText:
    def test_format_no_sections_shows_fresh_start(self):
        orc = MorningBriefingOrchestrator()
        briefing = orc.generate()
        text = orc.format_text(briefing)
        assert "Fresh start today!" in text

    def test_format_with_commitments_shows_section(self):
        orc = MorningBriefingOrchestrator()
        briefing = orc.generate(commitments=[{"text": "reply to boss", "due_by": "today"}])
        text = orc.format_text(briefing)
        assert "Pending Commitments" in text
        assert "reply to boss" in text

    def test_format_output_is_string(self):
        orc = MorningBriefingOrchestrator()
        briefing = orc.generate()
        assert isinstance(orc.format_text(briefing), str)

    def test_format_items_use_dash_prefix(self):
        orc = MorningBriefingOrchestrator()
        briefing = orc.generate(session_summary="Long day debugging.")
        text = orc.format_text(briefing)
        assert "- Long day debugging." in text

    def test_format_greeting_appears_first(self):
        orc = MorningBriefingOrchestrator()
        briefing = MorningBriefing(greeting="Good morning!", sections=[])
        text = orc.format_text(briefing)
        assert text.startswith("Good morning!")
