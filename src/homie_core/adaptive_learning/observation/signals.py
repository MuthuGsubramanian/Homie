"""Learning signal types for the observation pipeline."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SignalType(str, Enum):
    EXPLICIT = "explicit"      # Direct user feedback — high confidence
    IMPLICIT = "implicit"      # Inferred from behavior — medium confidence
    BEHAVIORAL = "behavioral"  # Background observation — low confidence


class SignalCategory(str, Enum):
    PREFERENCE = "preference"    # Response style preference
    ENGAGEMENT = "engagement"    # User engagement signal
    CONTEXT = "context"          # Context/environment signal
    PERFORMANCE = "performance"  # System performance signal


_CONFIDENCE_BY_TYPE = {
    SignalType.EXPLICIT: 0.9,
    SignalType.IMPLICIT: 0.5,
    SignalType.BEHAVIORAL: 0.3,
}


@dataclass
class LearningSignal:
    """A single learning observation from any source."""

    signal_type: SignalType
    category: SignalCategory
    source: str
    data: dict[str, Any]
    context: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    @property
    def confidence(self) -> float:
        """Signal confidence based on type."""
        return _CONFIDENCE_BY_TYPE.get(self.signal_type, 0.3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "category": self.category.value,
            "source": self.source,
            "data": self.data,
            "context": self.context,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }
