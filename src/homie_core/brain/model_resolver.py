"""ModelResolverMiddleware — dynamically selects the active model per turn."""
from __future__ import annotations

import logging
from typing import Any, Optional

from homie_core.config import ModelProfile, ModelTier
from homie_core.middleware.base import HomieMiddleware
from homie_core.middleware.hooks import HookRegistry, PipelineStage

logger = logging.getLogger(__name__)


class ModelResolverMiddleware(HomieMiddleware):
    """Middleware that chooses the best ModelProfile after classification.

    Resolution order (highest priority wins):
    1. ``model_override`` key in state (set by callers, consumed on use).
    2. CLASSIFIED hook — maps complexity string to ModelTier and picks best.
    3. Default: MEDIUM tier.
    """

    name = "model_resolver"
    order = 50

    # Maps complexity labels (from the classifier stage) to ModelTier values.
    TIER_MAP: dict[str, ModelTier] = {
        "trivial": ModelTier.SMALL,
        "simple": ModelTier.SMALL,
        "moderate": ModelTier.MEDIUM,
        "medium": ModelTier.MEDIUM,
        "complex": ModelTier.LARGE,
        "very_complex": ModelTier.LARGE,
    }

    def __init__(self, registry: Any, hooks: HookRegistry) -> None:
        """
        Args:
            registry: A ModelRegistry instance (or compatible duck-type).
            hooks:    The pipeline HookRegistry to register the CLASSIFIED listener on.
        """
        self._registry = registry
        self._state: dict = {}
        hooks.register(PipelineStage.CLASSIFIED, self._on_classified)

    # ------------------------------------------------------------------
    # HookRegistry callback
    # ------------------------------------------------------------------

    def _on_classified(self, stage: PipelineStage, complexity: Any) -> Any:
        """PRIMARY resolution point — called when the pipeline emits CLASSIFIED.

        Looks up the tier for *complexity*, finds the best available model,
        and stores it in ``self._state``.  Returns *complexity* unchanged so
        subsequent hooks continue to receive the original value.
        """
        tier = self.TIER_MAP.get(str(complexity).lower(), ModelTier.MEDIUM)
        profile: Optional[ModelProfile] = self._registry.best_for(tier)

        if profile is not None:
            self._state["active_model"] = profile.name
            self._state["active_model_location"] = profile.location
            logger.debug(
                "model_resolver: complexity=%s -> tier=%s -> model=%s (%s)",
                complexity, tier, profile.name, profile.location,
            )
        else:
            logger.debug(
                "model_resolver: no model found for tier=%s (complexity=%s)",
                tier, complexity,
            )

        return complexity  # pass-through

    # ------------------------------------------------------------------
    # Middleware lifecycle
    # ------------------------------------------------------------------

    def before_turn(self, message: str, state: dict) -> str:
        """Store a reference to the shared state dict and apply defaults/overrides.

        1. Saves *state* reference so ``_on_classified`` can write into it.
        2. Pops ``model_override`` if present and applies it immediately.
        3. Sets MEDIUM as the default if no model has been chosen yet.
        """
        self._state = state

        # Explicit override wins over everything else
        override: Optional[ModelProfile] = state.pop("model_override", None)
        if override is not None:
            state["active_model"] = override.name
            state["active_model_location"] = override.location
            logger.debug("model_resolver: override -> model=%s", override.name)
            return message

        # Fallback default: MEDIUM tier
        if "active_model" not in state:
            profile: Optional[ModelProfile] = self._registry.best_for(ModelTier.MEDIUM)
            if profile is not None:
                state["active_model"] = profile.name
                state["active_model_location"] = profile.location
            else:
                state.setdefault("active_model", None)
                state.setdefault("active_model_location", None)

        return message
