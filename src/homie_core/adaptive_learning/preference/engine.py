"""PreferenceEngine — learns response style from signals."""

import logging
from typing import Optional

from ..observation.signals import LearningSignal, SignalType
from ..storage import LearningStorage
from .profile import PreferenceLayer, PreferenceProfile, PreferenceResolver
from .prompt_builder import build_preference_prompt

logger = logging.getLogger(__name__)

# Direction mapping for numeric dimensions
_DIRECTION_VALUES = {"decrease": 0.0, "increase": 1.0}


class PreferenceEngine:
    """Learns and applies response style preferences."""

    def __init__(
        self,
        storage: LearningStorage,
        learning_rate_explicit: float = 0.3,
        learning_rate_implicit: float = 0.05,
    ) -> None:
        self._storage = storage
        self._lr_explicit = learning_rate_explicit
        self._lr_implicit = learning_rate_implicit
        self._resolver = PreferenceResolver()
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load saved profiles from storage."""
        for layer in PreferenceLayer:
            # Try loading known keys — global always exists
            if layer == PreferenceLayer.GLOBAL:
                data = self._storage.get_preference(layer.value, "default")
                if data:
                    self._resolver.set_profile(layer, "default", PreferenceProfile.from_dict(data))

    def on_signal(self, signal: LearningSignal) -> None:
        """Process a learning signal to update preferences."""
        data = signal.data
        dimension = data.get("dimension")
        if not dimension:
            # Implicit engagement signals — infer preference adjustments
            self._handle_implicit(signal)
            return

        # Determine learning rate
        lr = self._lr_explicit if signal.signal_type == SignalType.EXPLICIT else self._lr_implicit

        # Determine target value
        if "value" in data:
            # String dimension update (format, style)
            self._update_string_dimension(dimension, data["value"], signal.context)
            return

        direction = data.get("direction")
        if direction and direction in _DIRECTION_VALUES:
            target = _DIRECTION_VALUES[direction]
            self._update_numeric_dimension(dimension, target, lr, signal.context)

    def _handle_implicit(self, signal: LearningSignal) -> None:
        """Handle implicit engagement signals."""
        source = signal.source
        if source == "clarification_request":
            # Response wasn't clear — slightly increase verbosity
            self._update_numeric_dimension("verbosity", 0.7, self._lr_implicit, signal.context)
        elif source == "turn_complete":
            # Could track response length preferences over time
            pass

    def _update_numeric_dimension(
        self, dimension: str, target: float, lr: float, context: dict
    ) -> None:
        """Update a numeric preference dimension."""
        domain = context.get("topic")
        layer = PreferenceLayer.DOMAIN if domain else PreferenceLayer.GLOBAL
        key = domain or "default"

        profile = self._resolver.get_profile(layer, key)
        if profile is None:
            profile = PreferenceProfile()

        profile.update(dimension, target, learning_rate=lr)
        self._resolver.set_profile(layer, key, profile)
        self._storage.save_preference(layer.value, key, profile.to_dict())

    def _update_string_dimension(self, dimension: str, value: str, context: dict) -> None:
        """Update a string preference dimension."""
        domain = context.get("topic")
        layer = PreferenceLayer.DOMAIN if domain else PreferenceLayer.GLOBAL
        key = domain or "default"

        profile = self._resolver.get_profile(layer, key)
        if profile is None:
            profile = PreferenceProfile()

        if dimension == "format":
            profile.format_preference = value
        elif dimension == "style":
            profile.explanation_style = value
        profile.sample_count += 1
        profile.confidence = min(1.0, profile.sample_count / 50.0)

        self._resolver.set_profile(layer, key, profile)
        self._storage.save_preference(layer.value, key, profile.to_dict())

    def get_active_profile(
        self,
        domain: Optional[str] = None,
        project: Optional[str] = None,
        hour: Optional[int] = None,
    ) -> PreferenceProfile:
        """Get the resolved preference profile for a context."""
        return self._resolver.resolve(domain=domain, project=project, hour=hour)

    def get_prompt_layer(
        self,
        domain: Optional[str] = None,
        project: Optional[str] = None,
        hour: Optional[int] = None,
    ) -> str:
        """Get the preference prompt layer for injection into system prompt."""
        profile = self.get_active_profile(domain=domain, project=project, hour=hour)
        return build_preference_prompt(profile)
