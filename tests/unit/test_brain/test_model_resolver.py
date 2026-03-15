"""Tests for ModelResolverMiddleware."""
from __future__ import annotations

import pytest
from typing import Optional

from homie_core.config import ModelProfile, ModelTier
from homie_core.brain.model_resolver import ModelResolverMiddleware
from homie_core.middleware.hooks import HookRegistry, PipelineStage


# ---------------------------------------------------------------------------
# Fake registry
# ---------------------------------------------------------------------------

class FakeRegistry:
    """Minimal ModelRegistry stand-in backed by a plain dict."""

    def __init__(self, profiles: dict[ModelTier, ModelProfile]) -> None:
        self._profiles = profiles

    def best_for(self, tier: ModelTier) -> Optional[ModelProfile]:
        return self._profiles.get(tier)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def profiles() -> dict[ModelTier, ModelProfile]:
    return {
        ModelTier.SMALL: ModelProfile(
            name="small-model", tier=ModelTier.SMALL, location="local", priority=10
        ),
        ModelTier.MEDIUM: ModelProfile(
            name="medium-model", tier=ModelTier.MEDIUM, location="local", priority=10
        ),
        ModelTier.LARGE: ModelProfile(
            name="large-model", tier=ModelTier.LARGE, location="qubrid", priority=5
        ),
    }


@pytest.fixture
def hooks() -> HookRegistry:
    return HookRegistry()


@pytest.fixture
def resolver(profiles, hooks) -> ModelResolverMiddleware:
    return ModelResolverMiddleware(FakeRegistry(profiles), hooks)


# ---------------------------------------------------------------------------
# Middleware metadata
# ---------------------------------------------------------------------------

def test_name_and_order(resolver):
    assert resolver.name == "model_resolver"
    assert resolver.order == 50


# ---------------------------------------------------------------------------
# before_turn: default MEDIUM
# ---------------------------------------------------------------------------

def test_before_turn_sets_medium_default(resolver):
    state: dict = {}
    result = resolver.before_turn("hello", state)
    assert result == "hello"
    assert state["active_model"] == "medium-model"
    assert state["active_model_location"] == "local"


def test_before_turn_does_not_overwrite_existing_model(resolver):
    """If active_model is already set (e.g. by _on_classified), before_turn keeps it."""
    state: dict = {"active_model": "already-set", "active_model_location": "local"}
    resolver.before_turn("hi", state)
    assert state["active_model"] == "already-set"


# ---------------------------------------------------------------------------
# before_turn: explicit override
# ---------------------------------------------------------------------------

def test_before_turn_override_takes_precedence(resolver, profiles):
    override_profile = ModelProfile(
        name="override-model", tier=ModelTier.LARGE, location="lan", priority=99
    )
    state: dict = {"model_override": override_profile}
    resolver.before_turn("hi", state)

    assert state["active_model"] == "override-model"
    assert state["active_model_location"] == "lan"


def test_before_turn_override_is_consumed(resolver, profiles):
    """model_override must be popped from state after use."""
    override_profile = ModelProfile(
        name="override-model", tier=ModelTier.LARGE, location="lan", priority=99
    )
    state: dict = {"model_override": override_profile}
    resolver.before_turn("hi", state)

    assert "model_override" not in state


# ---------------------------------------------------------------------------
# _on_classified: trivial -> SMALL
# ---------------------------------------------------------------------------

def test_on_classified_trivial_selects_small(resolver):
    state: dict = {}
    resolver.before_turn("msg", state)  # bind state reference

    resolver._on_classified(PipelineStage.CLASSIFIED, "trivial")

    assert state["active_model"] == "small-model"
    assert state["active_model_location"] == "local"


def test_on_classified_simple_selects_small(resolver):
    state: dict = {}
    resolver.before_turn("msg", state)
    resolver._on_classified(PipelineStage.CLASSIFIED, "simple")
    assert state["active_model"] == "small-model"


# ---------------------------------------------------------------------------
# _on_classified: complex -> LARGE
# ---------------------------------------------------------------------------

def test_on_classified_complex_selects_large(resolver):
    state: dict = {}
    resolver.before_turn("msg", state)
    resolver._on_classified(PipelineStage.CLASSIFIED, "complex")

    assert state["active_model"] == "large-model"
    assert state["active_model_location"] == "qubrid"


def test_on_classified_very_complex_selects_large(resolver):
    state: dict = {}
    resolver.before_turn("msg", state)
    resolver._on_classified(PipelineStage.CLASSIFIED, "very_complex")
    assert state["active_model"] == "large-model"


# ---------------------------------------------------------------------------
# _on_classified: medium / moderate
# ---------------------------------------------------------------------------

def test_on_classified_moderate_selects_medium(resolver):
    state: dict = {}
    resolver.before_turn("msg", state)
    resolver._on_classified(PipelineStage.CLASSIFIED, "moderate")
    assert state["active_model"] == "medium-model"


def test_on_classified_medium_complexity_selects_medium(resolver):
    state: dict = {}
    resolver.before_turn("msg", state)
    resolver._on_classified(PipelineStage.CLASSIFIED, "medium")
    assert state["active_model"] == "medium-model"


# ---------------------------------------------------------------------------
# _on_classified returns complexity unchanged (pass-through)
# ---------------------------------------------------------------------------

def test_on_classified_returns_complexity_unchanged(resolver):
    state: dict = {}
    resolver.before_turn("msg", state)
    result = resolver._on_classified(PipelineStage.CLASSIFIED, "complex")
    assert result == "complex"


# ---------------------------------------------------------------------------
# Hook integration — HookRegistry fires _on_classified
# ---------------------------------------------------------------------------

def test_hook_registry_fires_on_classified(resolver, hooks):
    state: dict = {}
    resolver.before_turn("msg", state)

    # Emit through registry — should call resolver._on_classified
    hooks.emit(PipelineStage.CLASSIFIED, "trivial")

    assert state["active_model"] == "small-model"


# ---------------------------------------------------------------------------
# TIER_MAP completeness
# ---------------------------------------------------------------------------

def test_all_complexities_have_tier_mappings():
    expected_keys = {"trivial", "simple", "moderate", "medium", "complex", "very_complex"}
    assert expected_keys == set(ModelResolverMiddleware.TIER_MAP.keys())


def test_tier_map_values_are_model_tiers():
    for complexity, tier in ModelResolverMiddleware.TIER_MAP.items():
        assert isinstance(tier, ModelTier), (
            f"TIER_MAP['{complexity}'] should be a ModelTier, got {type(tier)}"
        )


# ---------------------------------------------------------------------------
# Edge: empty registry
# ---------------------------------------------------------------------------

def test_before_turn_with_empty_registry(hooks):
    empty_registry = FakeRegistry({})
    resolver = ModelResolverMiddleware(empty_registry, hooks)
    state: dict = {}
    result = resolver.before_turn("hi", state)

    assert result == "hi"
    assert state.get("active_model") is None


def test_on_classified_with_empty_registry(hooks):
    empty_registry = FakeRegistry({})
    resolver = ModelResolverMiddleware(empty_registry, hooks)
    state: dict = {}
    resolver.before_turn("hi", state)

    # Should not raise even when no models are available
    returned = resolver._on_classified(PipelineStage.CLASSIFIED, "complex")
    assert returned == "complex"
    # active_model stays None / unchanged
    assert state.get("active_model") is None


# ---------------------------------------------------------------------------
# Case-insensitive complexity mapping
# ---------------------------------------------------------------------------

def test_on_classified_case_insensitive(resolver):
    state: dict = {}
    resolver.before_turn("msg", state)
    resolver._on_classified(PipelineStage.CLASSIFIED, "TRIVIAL")
    assert state["active_model"] == "small-model"


def test_on_classified_unknown_complexity_defaults_to_medium(resolver):
    """Unknown complexity strings should fall back to MEDIUM tier."""
    state: dict = {}
    resolver.before_turn("msg", state)
    resolver._on_classified(PipelineStage.CLASSIFIED, "unknownXYZ")
    assert state["active_model"] == "medium-model"
