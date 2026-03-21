import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.observation.learning_middleware import LearningMiddleware
from homie_core.adaptive_learning.observation.signals import SignalType, SignalCategory


class TestLearningMiddleware:
    def _make_mw(self, stream=None):
        stream = stream or MagicMock()
        return LearningMiddleware(observation_stream=stream)

    def test_has_correct_name(self):
        mw = self._make_mw()
        assert mw.name == "learning"

    def test_before_turn_records_timestamp(self):
        mw = self._make_mw()
        mw.before_turn("hello", {})
        assert mw._turn_start_time is not None

    def test_after_turn_emits_engagement_signal(self):
        stream = MagicMock()
        mw = LearningMiddleware(observation_stream=stream)
        mw.before_turn("hello", {})
        result = mw.after_turn("response here", {"topic": "coding"})
        assert result == "response here"
        stream.emit.assert_called()
        signal = stream.emit.call_args[0][0]
        assert signal.category == SignalCategory.ENGAGEMENT

    def test_detects_explicit_preference(self):
        stream = MagicMock()
        mw = LearningMiddleware(observation_stream=stream)
        mw.before_turn("be more concise please", {})
        # Check that an explicit preference signal was emitted
        calls = [c[0][0] for c in stream.emit.call_args_list]
        explicit = [c for c in calls if c.signal_type == SignalType.EXPLICIT]
        assert len(explicit) >= 1

    def test_detects_clarification_request(self):
        stream = MagicMock()
        mw = LearningMiddleware(observation_stream=stream)
        # Simulate previous turn existed
        mw._last_response = "some previous response"
        mw.before_turn("what do you mean?", {})
        calls = [c[0][0] for c in stream.emit.call_args_list]
        implicit = [c for c in calls if c.signal_type == SignalType.IMPLICIT]
        assert len(implicit) >= 1

    def test_passes_through_response_unmodified(self):
        mw = self._make_mw()
        mw.before_turn("hi", {})
        assert mw.after_turn("hello back", {}) == "hello back"
