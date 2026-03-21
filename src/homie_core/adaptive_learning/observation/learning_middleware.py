"""LearningMiddleware — captures implicit and explicit signals from each turn."""

import re
import time
from typing import Optional

from homie_core.middleware.base import HomieMiddleware

from .signals import LearningSignal, SignalCategory, SignalType
from .stream import ObservationStream

# Patterns for explicit preference detection
_EXPLICIT_PATTERNS = [
    (r"\b(be\s+)?more\s+(concise|brief|short)", {"dimension": "verbosity", "direction": "decrease"}),
    (r"\b(be\s+)?more\s+(detailed|verbose|thorough)", {"dimension": "verbosity", "direction": "increase"}),
    (r"\b(be\s+)?more\s+(formal|professional)", {"dimension": "formality", "direction": "increase"}),
    (r"\b(be\s+)?more\s+(casual|informal)", {"dimension": "formality", "direction": "decrease"}),
    (r"\b(be\s+)?more\s+(technical|advanced)", {"dimension": "technical_depth", "direction": "increase"}),
    (r"\b(be\s+)?more\s+simple|simplify", {"dimension": "technical_depth", "direction": "decrease"}),
    (r"\buse\s+bullet\s*points?", {"dimension": "format", "value": "bullets"}),
    (r"\bshow\s+(me\s+)?code\s+first", {"dimension": "format", "value": "code_first"}),
    (r"\bskip\s+the\s+explanation", {"dimension": "depth", "direction": "decrease"}),
]

# Patterns for implicit clarification detection
_CLARIFICATION_PATTERNS = [
    r"\bwhat\s+do\s+you\s+mean",
    r"\bcan\s+you\s+explain",
    r"\bi\s+don'?t\s+understand",
    r"\bwhat\s+does\s+that\s+mean",
    r"\bclarify",
]


class LearningMiddleware(HomieMiddleware):
    """Captures learning signals from each conversation turn."""

    name = "learning"
    order = 900  # Run late — after other middleware

    def __init__(self, observation_stream: ObservationStream) -> None:
        self._stream = observation_stream
        self._turn_start_time: Optional[float] = None
        self._last_response: Optional[str] = None

    def before_turn(self, message: str, state: dict) -> Optional[str]:
        """Record turn start and detect explicit/implicit signals from user message."""
        self._turn_start_time = time.time()

        # Detect explicit preferences
        msg_lower = message.lower()
        for pattern, data in _EXPLICIT_PATTERNS:
            if re.search(pattern, msg_lower):
                self._stream.emit(LearningSignal(
                    signal_type=SignalType.EXPLICIT,
                    category=SignalCategory.PREFERENCE,
                    source="user_message",
                    data=data,
                    context=state.copy() if state else {},
                ))

        # Detect clarification requests (implicit signal)
        if self._last_response:
            for pattern in _CLARIFICATION_PATTERNS:
                if re.search(pattern, msg_lower):
                    self._stream.emit(LearningSignal(
                        signal_type=SignalType.IMPLICIT,
                        category=SignalCategory.ENGAGEMENT,
                        source="clarification_request",
                        data={"previous_response_len": len(self._last_response)},
                        context=state.copy() if state else {},
                    ))
                    break

        return message

    def after_turn(self, response: str, state: dict) -> str:
        """Emit engagement signal after response is generated."""
        elapsed_ms = 0.0
        if self._turn_start_time:
            elapsed_ms = (time.time() - self._turn_start_time) * 1000

        self._stream.emit(LearningSignal(
            signal_type=SignalType.IMPLICIT,
            category=SignalCategory.ENGAGEMENT,
            source="turn_complete",
            data={
                "response_length": len(response),
                "generation_time_ms": elapsed_ms,
            },
            context=state.copy() if state else {},
        ))

        self._last_response = response
        return response
