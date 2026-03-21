"""PreferenceProfile — multi-dimensional response style preferences with layering."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PreferenceLayer(str, Enum):
    GLOBAL = "global"
    DOMAIN = "domain"
    PROJECT = "project"
    TEMPORAL = "temporal"


@dataclass
class PreferenceProfile:
    """Multi-dimensional profile describing preferred response style."""

    verbosity: float = 0.5         # 0=terse, 1=verbose
    formality: float = 0.5         # 0=casual, 1=formal
    technical_depth: float = 0.7   # 0=simple, 1=expert
    format_preference: str = "mixed"       # prose, bullets, code_first, mixed
    explanation_style: str = "top_down"    # bottom_up, top_down, example_first
    confidence: float = 0.0
    sample_count: int = 0

    def update(self, dimension: str, target_value: float, learning_rate: float = 0.1) -> None:
        """Update a numeric dimension toward target_value using EMA."""
        if not hasattr(self, dimension):
            return
        current = getattr(self, dimension)
        if not isinstance(current, (int, float)):
            return
        new_value = learning_rate * target_value + (1 - learning_rate) * current
        new_value = max(0.0, min(1.0, new_value))
        setattr(self, dimension, new_value)
        self.sample_count += 1
        self.confidence = min(1.0, self.sample_count / 50.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verbosity": self.verbosity,
            "formality": self.formality,
            "technical_depth": self.technical_depth,
            "format_preference": self.format_preference,
            "explanation_style": self.explanation_style,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreferenceProfile":
        return cls(
            verbosity=data.get("verbosity", 0.5),
            formality=data.get("formality", 0.5),
            technical_depth=data.get("technical_depth", 0.7),
            format_preference=data.get("format_preference", "mixed"),
            explanation_style=data.get("explanation_style", "top_down"),
            confidence=data.get("confidence", 0.0),
            sample_count=data.get("sample_count", 0),
        )


class PreferenceResolver:
    """Resolves the active preference profile from layered profiles."""

    def __init__(self) -> None:
        # {(layer, key): PreferenceProfile}
        self._profiles: dict[tuple[PreferenceLayer, str], PreferenceProfile] = {}

    def set_profile(self, layer: PreferenceLayer, key: str, profile: PreferenceProfile) -> None:
        self._profiles[(layer, key)] = profile

    def get_profile(self, layer: PreferenceLayer, key: str) -> Optional[PreferenceProfile]:
        return self._profiles.get((layer, key))

    def resolve(
        self,
        domain: Optional[str] = None,
        project: Optional[str] = None,
        hour: Optional[int] = None,
    ) -> PreferenceProfile:
        """Resolve the active profile. Most specific layer wins."""
        # Resolution order: temporal -> project -> domain -> global
        if hour is not None:
            temporal_key = f"hour_{hour}"
            if (PreferenceLayer.TEMPORAL, temporal_key) in self._profiles:
                return self._profiles[(PreferenceLayer.TEMPORAL, temporal_key)]

        if project and (PreferenceLayer.PROJECT, project) in self._profiles:
            return self._profiles[(PreferenceLayer.PROJECT, project)]

        if domain and (PreferenceLayer.DOMAIN, domain) in self._profiles:
            return self._profiles[(PreferenceLayer.DOMAIN, domain)]

        return self._profiles.get(
            (PreferenceLayer.GLOBAL, "default"),
            PreferenceProfile(),
        )
