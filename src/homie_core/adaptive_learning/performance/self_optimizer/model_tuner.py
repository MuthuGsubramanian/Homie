"""Model tuner — selects optimal per-call parameters based on query type and learned profiles."""

from typing import Any, Optional

from .profiler import OptimizationProfiler

# Default parameters by complexity tier
_TIER_DEFAULTS = {
    "trivial":  {"max_tokens": 256,  "temperature": 0.8},
    "simple":   {"max_tokens": 384,  "temperature": 0.7},
    "moderate": {"max_tokens": 768,  "temperature": 0.7},
    "complex":  {"max_tokens": 1536, "temperature": 0.6},
    "deep":     {"max_tokens": 3072, "temperature": 0.5},
}

# Query hint overrides for temperature
_HINT_TEMPERATURE = {
    "coding": 0.3,
    "factual": 0.3,
    "creative": 0.9,
    "chat": 0.7,
}

_MIN_PROFILE_SAMPLES = 5  # need at least this many observations before trusting profile


class ModelTuner:
    """Selects optimal inference parameters per query."""

    def __init__(self, profiler: OptimizationProfiler) -> None:
        self._profiler = profiler

    def select_parameters(
        self,
        complexity: str,
        query_hint: Optional[str] = None,
    ) -> dict[str, Any]:
        """Select optimal temperature and max_tokens for this query."""
        # Start with tier defaults
        defaults = _TIER_DEFAULTS.get(complexity, _TIER_DEFAULTS["moderate"])
        temperature = defaults["temperature"]
        max_tokens = defaults["max_tokens"]

        # Apply query hint
        if query_hint and query_hint in _HINT_TEMPERATURE:
            temperature = _HINT_TEMPERATURE[query_hint]

        # Check for learned profile
        profile_key = query_hint or complexity
        profile = self._profiler.get_profile(profile_key)
        if profile and profile.sample_count >= _MIN_PROFILE_SAMPLES:
            temperature = profile.temperature
            max_tokens = profile.max_tokens
            # Adjust max_tokens based on learned response length
            if profile.avg_response_tokens > 0:
                max_tokens = max(64, int(profile.avg_response_tokens * 1.5))

        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def record_result(
        self,
        query_type: str,
        temperature: float,
        max_tokens: int,
        response_tokens: float,
        latency_ms: float,
    ) -> None:
        """Record an inference result to update the profile."""
        self._profiler.record_observation(
            query_type=query_type,
            temperature_used=temperature,
            max_tokens_used=max_tokens,
            response_tokens=response_tokens,
            latency_ms=latency_ms,
        )
