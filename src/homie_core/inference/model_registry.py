"""ModelRegistry — scans and tracks available inference models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from homie_core.config import HomieConfig, ModelProfile, ModelTier

logger = logging.getLogger(__name__)

# Byte thresholds for automatic tier assignment
_SMALL_THRESHOLD = 4 * 1024 ** 3   # < 4 GB  -> SMALL
_MEDIUM_THRESHOLD = 16 * 1024 ** 3  # < 16 GB -> MEDIUM  (else LARGE)


class ModelRegistry:
    """Registry of available ModelProfile objects, populated by scanning sources."""

    def __init__(self, config: HomieConfig) -> None:
        self._config = config
        self._profiles: dict[str, ModelProfile] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Clear all profiles and re-scan every source."""
        self._profiles.clear()
        self._scan_local()
        self._scan_lan()
        self._scan_qubrid()

    def available(self, tier: Optional[ModelTier] = None) -> list[ModelProfile]:
        """Return all profiles sorted by priority (highest first).

        Args:
            tier: If provided, filter to profiles matching this tier.
        """
        profiles = list(self._profiles.values())
        if tier is not None:
            profiles = [p for p in profiles if p.tier == tier]
        return sorted(profiles, key=lambda p: p.priority, reverse=True)

    def best_for(self, tier: ModelTier) -> Optional[ModelProfile]:
        """Return the highest-priority profile for *tier*, or None."""
        candidates = self.available(tier)
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add(self, profile: ModelProfile) -> None:
        """Register a profile, keyed by name."""
        self._profiles[profile.name] = profile

    def _scan_local(self) -> None:
        """Scan ``<storage.path>/models/`` for .gguf files and register them."""
        models_dir = Path(self._config.storage.path) / "models"
        if not models_dir.is_dir():
            return

        for gguf_path in models_dir.glob("*.gguf"):
            try:
                size = gguf_path.stat().st_size
            except OSError:
                logger.warning("Cannot stat model file: %s", gguf_path)
                continue

            if size < _SMALL_THRESHOLD:
                tier = ModelTier.SMALL
            elif size < _MEDIUM_THRESHOLD:
                tier = ModelTier.MEDIUM
            else:
                tier = ModelTier.LARGE

            profile = ModelProfile(
                name=gguf_path.stem,
                tier=tier,
                location="local",
                priority=10,
            )
            self._add(profile)
            logger.debug("Registered local model: %s (tier=%s)", profile.name, tier)

    def _scan_lan(self) -> None:
        """Stub — LAN scanning not yet implemented."""
        pass

    def _scan_qubrid(self) -> None:
        """Register Qubrid cloud model if an API key is configured."""
        api_key = self._config.llm.api_key
        if not api_key:
            return

        try:
            from homie_core.inference.qubrid import QubridClient  # noqa: F401
        except ImportError:
            logger.warning("QubridClient not available; skipping Qubrid scan.")
            return

        try:
            model_name = self._config.inference.qubrid.model
            profile = ModelProfile(
                name=model_name,
                tier=ModelTier.LARGE,
                location="qubrid",
                priority=5,
                context_length=32768,
            )
            self._add(profile)
            logger.debug("Registered Qubrid model: %s", model_name)
        except Exception as exc:
            logger.warning("Failed to register Qubrid model: %s", exc)
