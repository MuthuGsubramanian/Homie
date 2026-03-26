"""Tests for tool executor."""
import pytest
from unittest.mock import MagicMock
from homie_core.brain.tool_executor import ToolExecutor
from homie_core.brain.tool_registry import ToolResult


class TestToolExecutor:
    def _mock_registry(self, tools=None):
        registry = MagicMock()
        available = tools or {"email_unread", "email_summary", "remember", "recall", "git_status"}
        registry.has_tool.side_effect = lambda name: name in available
        registry.execute.return_value = ToolResult(
            tool_name="mock", success=True, output="Tool result here"
        )
        return registry

    def test_can_execute_with_matching_intent(self):
        registry = self._mock_registry()
        executor = ToolExecutor(tool_registry=registry)
        assert executor.can_execute("Check my emails") is True

    def test_cannot_execute_unknown_intent(self):
        registry = self._mock_registry()
        executor = ToolExecutor(tool_registry=registry)
        assert executor.can_execute("Tell me a joke") is False

    def test_cannot_execute_without_registry(self):
        executor = ToolExecutor(tool_registry=None)
        assert executor.can_execute("Check my emails") is False

    def test_execute_check_email(self):
        registry = self._mock_registry()
        executor = ToolExecutor(tool_registry=registry)
        result = executor.execute("Check my emails")
        assert result is not None
        assert result["intent"] == "check_email"
        assert len(result["results"]) > 0

    def test_execute_remember_extracts_fact(self):
        registry = self._mock_registry()
        executor = ToolExecutor(tool_registry=registry)
        result = executor.execute("Remember that I prefer dark mode")
        assert result is not None
        assert result["intent"] == "remember_fact"

    def test_draft_email_needs_confirmation(self):
        registry = self._mock_registry({"email_draft"})
        executor = ToolExecutor(tool_registry=registry)
        result = executor.execute("Send an email to John about the deadline")
        assert result is not None
        assert result["needs_confirmation"] is True

    def test_execute_git_status(self):
        registry = self._mock_registry()
        executor = ToolExecutor(tool_registry=registry)
        result = executor.execute("What's the git status?")
        assert result is not None
        assert result["intent"] == "git_status"

    def test_no_match_returns_none(self):
        executor = ToolExecutor(tool_registry=self._mock_registry())
        result = executor.execute("Hello there!")
        assert result is None

    def test_execute_confirmed(self):
        registry = self._mock_registry({"email_draft"})
        executor = ToolExecutor(tool_registry=registry)
        result = executor.execute_confirmed("draft_email", {"to": "John", "subject": "Deadline", "body": "Hi"})
        assert result["formatted"]

    def test_handles_tool_error_gracefully(self):
        registry = self._mock_registry()
        registry.execute.side_effect = RuntimeError("Tool crashed")
        executor = ToolExecutor(tool_registry=registry)
        result = executor.execute("Check my emails")
        assert result is not None
        failed = [r for r in result["results"] if not r["success"]]
        assert len(failed) > 0

    def test_extract_params_email_to(self):
        executor = ToolExecutor(tool_registry=self._mock_registry({"email_draft"}))
        result = executor.execute("Send an email to John about project status")
        assert result["params"]["to"] == "John"
        assert "project" in result["params"].get("subject", "")
