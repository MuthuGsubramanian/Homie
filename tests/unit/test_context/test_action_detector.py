"""Tests for action_detector module."""
from __future__ import annotations
import time

import pytest

from homie_core.context.action_detector import ActionDetector, ActionEvent


class TestActionDetectorIdle:
    def test_empty_detector_returns_idle(self):
        detector = ActionDetector()
        assert detector.current_activity() == "idle"

    def test_event_count_zero_when_empty(self):
        detector = ActionDetector()
        assert detector.event_count == 0


class TestActionDetectorCoding:
    def test_vscode_returns_coding(self):
        detector = ActionDetector()
        detector.push("code.exe", "main.py - VSCode", time.time())
        assert detector.current_activity() == "coding"

    def test_pycharm_returns_coding(self):
        detector = ActionDetector()
        detector.push("pycharm64.exe", "Homie - PyCharm", time.time())
        assert detector.current_activity() == "coding"

    def test_nvim_returns_coding(self):
        detector = ActionDetector()
        detector.push("nvim", "router.py", time.time())
        assert detector.current_activity() == "coding"

    def test_vim_returns_coding(self):
        detector = ActionDetector()
        detector.push("vim", "test.py", time.time())
        assert detector.current_activity() == "coding"

    def test_idea_returns_coding(self):
        detector = ActionDetector()
        detector.push("idea64.exe", "MyProject - IntelliJ", time.time())
        assert detector.current_activity() == "coding"


class TestActionDetectorBrowsing:
    def test_chrome_returns_browsing(self):
        detector = ActionDetector()
        detector.push("chrome.exe", "Google - Chrome", time.time())
        assert detector.current_activity() == "browsing"

    def test_firefox_returns_browsing(self):
        detector = ActionDetector()
        detector.push("firefox.exe", "Mozilla Firefox", time.time())
        assert detector.current_activity() == "browsing"

    def test_safari_returns_browsing(self):
        detector = ActionDetector()
        detector.push("safari", "Safari", time.time())
        assert detector.current_activity() == "browsing"

    def test_msedge_returns_browsing(self):
        detector = ActionDetector()
        detector.push("msedge.exe", "Microsoft Edge", time.time())
        assert detector.current_activity() == "browsing"


class TestActionDetectorMeeting:
    def test_zoom_returns_meeting(self):
        detector = ActionDetector()
        detector.push("zoom.exe", "Zoom Meeting", time.time())
        assert detector.current_activity() == "meeting"

    def test_teams_returns_meeting(self):
        detector = ActionDetector()
        detector.push("teams.exe", "Microsoft Teams", time.time())
        assert detector.current_activity() == "meeting"

    def test_slack_returns_meeting(self):
        detector = ActionDetector()
        detector.push("slack.exe", "Slack - General", time.time())
        assert detector.current_activity() == "meeting"

    def test_meeting_takes_priority_over_coding(self):
        """Meeting should take priority over coding apps in the same window."""
        detector = ActionDetector()
        detector.push("code.exe", "editor", time.time())
        detector.push("zoom.exe", "Zoom Meeting", time.time())
        assert detector.current_activity() == "meeting"


class TestActionDetectorWriting:
    def test_notion_in_title_returns_writing(self):
        detector = ActionDetector()
        detector.push("chrome.exe", "Notion – Project Notes", time.time())
        assert detector.current_activity() == "writing"

    def test_obsidian_in_title_returns_writing(self):
        detector = ActionDetector()
        detector.push("obsidian", "Daily Note - Obsidian", time.time())
        assert detector.current_activity() == "writing"

    def test_draft_in_title_returns_writing(self):
        detector = ActionDetector()
        detector.push("chrome.exe", "Draft - My Blog Post", time.time())
        assert detector.current_activity() == "writing"

    def test_word_in_title_returns_writing(self):
        detector = ActionDetector()
        detector.push("chrome.exe", "word document - Google Docs", time.time())
        assert detector.current_activity() == "writing"

    def test_coding_takes_priority_over_writing(self):
        """Coding app should override writing keyword in title."""
        detector = ActionDetector()
        detector.push("code.exe", "draft README.md - VSCode", time.time())
        assert detector.current_activity() == "coding"


class TestActionDetectorWorking:
    def test_unknown_app_returns_working(self):
        detector = ActionDetector()
        detector.push("excel.exe", "Budget Q4", time.time())
        assert detector.current_activity() == "working"


class TestActionDetectorWindowSize:
    def test_window_size_limits_events(self):
        detector = ActionDetector(window_size=5)
        for i in range(10):
            detector.push("some.exe", "title", float(i))
        assert detector.event_count == 5

    def test_push_increments_event_count(self):
        detector = ActionDetector()
        assert detector.event_count == 0
        detector.push("app.exe", "title", 1.0)
        assert detector.event_count == 1

    def test_recent_events_override_old_for_activity(self):
        """The last 5 events determine activity, not older ones."""
        detector = ActionDetector(window_size=30)
        # Push 6 coding events
        for i in range(6):
            detector.push("code.exe", "editor", float(i))
        # Override with 5 meeting events
        for i in range(5):
            detector.push("zoom.exe", "Zoom", float(10 + i))
        assert detector.current_activity() == "meeting"

    def test_app_case_insensitive(self):
        """App names should match case-insensitively."""
        detector = ActionDetector()
        detector.push("CODE.EXE", "editor", 1.0)
        assert detector.current_activity() == "coding"
