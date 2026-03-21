# tests/unit/adaptive_learning/test_preference_engine.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.preference.engine import PreferenceEngine
from homie_core.adaptive_learning.observation.signals import (
    LearningSignal,
    SignalType,
    SignalCategory,
)


class TestPreferenceEngine:
    def _make_engine(self, storage=None):
        storage = storage or MagicMock()
        storage.get_preference.return_value = None
        return PreferenceEngine(
            storage=storage,
            learning_rate_explicit=0.3,
            learning_rate_implicit=0.05,
        )

    def test_handles_explicit_verbosity_decrease(self):
        engine = self._make_engine()
        sig = LearningSignal(
            signal_type=SignalType.EXPLICIT,
            category=SignalCategory.PREFERENCE,
            source="user_message",
            data={"dimension": "verbosity", "direction": "decrease"},
            context={},
        )
        engine.on_signal(sig)
        profile = engine.get_active_profile()
        assert profile.verbosity < 0.5  # should have decreased from default

    def test_handles_explicit_format_change(self):
        engine = self._make_engine()
        sig = LearningSignal(
            signal_type=SignalType.EXPLICIT,
            category=SignalCategory.PREFERENCE,
            source="user_message",
            data={"dimension": "format", "value": "bullets"},
            context={},
        )
        engine.on_signal(sig)
        profile = engine.get_active_profile()
        assert profile.format_preference == "bullets"

    def test_implicit_signal_learns_slowly(self):
        engine = self._make_engine()
        sig = LearningSignal(
            signal_type=SignalType.IMPLICIT,
            category=SignalCategory.ENGAGEMENT,
            source="clarification_request",
            data={"previous_response_len": 500},
            context={},
        )
        engine.on_signal(sig)
        profile = engine.get_active_profile()
        # Implicit signal should barely move the needle
        assert 0.45 < profile.verbosity < 0.55

    def test_domain_context_creates_domain_profile(self):
        engine = self._make_engine()
        sig = LearningSignal(
            signal_type=SignalType.EXPLICIT,
            category=SignalCategory.PREFERENCE,
            source="user_message",
            data={"dimension": "verbosity", "direction": "decrease"},
            context={"topic": "coding"},
        )
        engine.on_signal(sig)
        profile = engine.get_active_profile(domain="coding")
        assert profile.verbosity < 0.5

    def test_get_prompt_layer(self):
        engine = self._make_engine()
        # Feed enough signals to build confidence
        for _ in range(15):
            engine.on_signal(LearningSignal(
                signal_type=SignalType.EXPLICIT,
                category=SignalCategory.PREFERENCE,
                source="user",
                data={"dimension": "verbosity", "direction": "decrease"},
                context={},
            ))
        prompt = engine.get_prompt_layer()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_saves_to_storage(self):
        storage = MagicMock()
        storage.get_preference.return_value = None
        engine = PreferenceEngine(storage=storage, learning_rate_explicit=0.3, learning_rate_implicit=0.05)
        engine.on_signal(LearningSignal(
            signal_type=SignalType.EXPLICIT,
            category=SignalCategory.PREFERENCE,
            source="user",
            data={"dimension": "verbosity", "direction": "decrease"},
            context={},
        ))
        storage.save_preference.assert_called()
