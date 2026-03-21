import time
import pytest
from homie_core.adaptive_learning.observation.signals import (
    LearningSignal,
    SignalType,
    SignalCategory,
)


class TestLearningSignal:
    def test_creation(self):
        sig = LearningSignal(
            signal_type=SignalType.EXPLICIT,
            category=SignalCategory.PREFERENCE,
            source="user_feedback",
            data={"dimension": "verbosity", "direction": "decrease"},
            context={"topic": "coding"},
        )
        assert sig.signal_type == SignalType.EXPLICIT
        assert sig.category == SignalCategory.PREFERENCE
        assert sig.timestamp > 0

    def test_to_dict(self):
        sig = LearningSignal(
            signal_type=SignalType.IMPLICIT,
            category=SignalCategory.ENGAGEMENT,
            source="response_timing",
            data={"response_time_ms": 1500},
            context={},
        )
        d = sig.to_dict()
        assert d["signal_type"] == "explicit" or d["signal_type"] == "implicit"
        assert "timestamp" in d

    def test_confidence_by_type(self):
        explicit = LearningSignal(
            signal_type=SignalType.EXPLICIT,
            category=SignalCategory.PREFERENCE,
            source="user",
            data={},
            context={},
        )
        implicit = LearningSignal(
            signal_type=SignalType.IMPLICIT,
            category=SignalCategory.ENGAGEMENT,
            source="timing",
            data={},
            context={},
        )
        behavioral = LearningSignal(
            signal_type=SignalType.BEHAVIORAL,
            category=SignalCategory.CONTEXT,
            source="app",
            data={},
            context={},
        )
        assert explicit.confidence == 0.9
        assert implicit.confidence == 0.5
        assert behavioral.confidence == 0.3


class TestSignalType:
    def test_enum_values(self):
        assert SignalType.EXPLICIT.value == "explicit"
        assert SignalType.IMPLICIT.value == "implicit"
        assert SignalType.BEHAVIORAL.value == "behavioral"


class TestSignalCategory:
    def test_enum_values(self):
        assert SignalCategory.PREFERENCE.value == "preference"
        assert SignalCategory.ENGAGEMENT.value == "engagement"
        assert SignalCategory.CONTEXT.value == "context"
        assert SignalCategory.PERFORMANCE.value == "performance"
