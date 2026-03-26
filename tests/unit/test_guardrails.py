"""Tests for inference guardrails."""
import pytest
from unittest.mock import MagicMock, call
from homie_core.inference.guardrails import (
    is_blank_response, rephrase_for_retry, guarded_inference,
)


class TestIsBlankResponse:
    def test_empty_string(self):
        assert is_blank_response("") is True

    def test_none(self):
        assert is_blank_response(None) is True

    def test_short_response(self):
        assert is_blank_response("OK") is True

    def test_normal_response(self):
        assert is_blank_response("Here is a helpful response about your question.") is False

    def test_repetitive_response(self):
        assert is_blank_response("a reminder " * 50) is True

    def test_whitespace_only(self):
        assert is_blank_response("   \n\n   ") is True


class TestRephraseForRetry:
    def test_check_command(self):
        result = rephrase_for_retry("Check my emails and summarize important ones")
        assert "help" in result.lower()

    def test_on_my_plate(self):
        result = rephrase_for_retry("What do I have on my plate today?")
        assert "focus" in result.lower()

    def test_schedule_query(self):
        result = rephrase_for_retry("Show me my schedule for today")
        assert "schedule" in result.lower() or "upcoming" in result.lower()

    def test_generic_command(self):
        result = rephrase_for_retry("Do something complex")
        assert "help" in result.lower() or "need" in result.lower()


class TestGuardedInference:
    def test_passes_through_good_response(self):
        fn = MagicMock(return_value="A great response")
        messages = [{"role": "user", "content": "Hello"}]
        result = guarded_inference(fn, messages)
        assert result == "A great response"
        fn.assert_called_once()

    def test_retries_on_blank(self):
        fn = MagicMock(side_effect=["", "This is a good response on the retry attempt."])
        messages = [{"role": "user", "content": "What do I have on my plate today?"}]
        result = guarded_inference(fn, messages)
        assert "good response" in result
        assert fn.call_count == 2

    def test_returns_fallback_after_all_retries_fail(self):
        fn = MagicMock(return_value="")
        messages = [{"role": "user", "content": "Check emails and send reminder"}]
        result = guarded_inference(fn, messages, max_retries=2)
        assert "rephras" in result.lower()
        assert fn.call_count == 3  # 1 original + 2 retries

    def test_rephrases_user_message_on_retry(self):
        calls = []
        def capture_fn(**kwargs):
            msgs = kwargs.get("messages", [])
            user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
            calls.append(user_msgs[-1] if user_msgs else "")
            return "" if len(calls) == 1 else "This is the rephrased response with enough length."

        messages = [
            {"role": "system", "content": "You are Homie"},
            {"role": "user", "content": "What do I have on my plate today?"},
        ]
        result = guarded_inference(capture_fn, messages)
        assert "rephrased response" in result
        # First call: original, second call: rephrased
        assert calls[0] == "What do I have on my plate today?"
        assert calls[1] != calls[0]  # Should be rephrased
