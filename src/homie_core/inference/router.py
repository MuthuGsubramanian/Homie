"""Unified inference router — local -> LAN -> Qubrid."""
from __future__ import annotations

import logging
from typing import Iterator, Optional

from homie_core.config import HomieConfig
from homie_core.model.engine import ModelEngine

logger = logging.getLogger(__name__)

_TIER_TIMEOUTS = {
    "small": 8,
    "medium": 25,
    "large": 90,
}

_FALLBACK_BANNER = (
    "No local model found! Using Homie's intelligence until local model is setup!"
)


class InferenceRouter:
    """Routes inference requests through the priority chain."""

    def __init__(
        self,
        config: HomieConfig,
        model_engine: ModelEngine,
        qubrid_api_key: str = "",
        lan_device_id: str = "",
        lan_device_name: str = "",
    ):
        self._config = config
        self._engine = model_engine
        self._qubrid = None
        self._lan = None
        self._priority = config.inference.priority

        if qubrid_api_key and config.inference.qubrid.enabled:
            from homie_core.inference.qubrid import QubridClient
            self._qubrid = QubridClient(
                api_key=qubrid_api_key,
                model=config.inference.qubrid.model,
                base_url=config.inference.qubrid.base_url,
                timeout=config.inference.qubrid.timeout,
            )
            self._qubrid.check_available()

        if lan_device_id and "lan" in self._priority:
            try:
                from homie_core.inference.lan_client import LANInferenceClient
                self._lan = LANInferenceClient(
                    device_id=lan_device_id,
                    device_name=lan_device_name,
                )
                self._lan.start()
            except Exception as exc:
                logger.warning("Failed to start LAN inference client: %s", exc)

    @property
    def active_source(self) -> str:
        for source in self._priority:
            if source == "local" and self._engine.is_loaded:
                return "Local"
            if source == "lan" and self._lan and self._lan.is_available:
                return "LAN Peer"
            if source == "qubrid" and self._qubrid and self._qubrid.is_available:
                return "Homie Intelligence (Cloud)"
        return "None"

    @property
    def fallback_banner(self) -> Optional[str]:
        if self._engine.is_loaded:
            return None
        if self._qubrid and self._qubrid.is_available:
            return _FALLBACK_BANNER
        return None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        timeout: int = 120,
        model: Optional[str] = None,
        preferred_location: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> str:
        if tier is not None:
            timeout = _TIER_TIMEOUTS.get(tier, timeout)
        logger.debug(
            "InferenceRouter.generate — model hint=%r, preferred_location=%r, tier=%r",
            model,
            preferred_location,
            tier,
        )
        errors: list[str] = []
        for source in self._priority:
            try:
                if source == "local" and self._engine.is_loaded:
                    return self._engine.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop, timeout=timeout,
                    )
                if source == "lan" and self._lan and self._lan.is_available:
                    return self._lan.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                if source == "qubrid" and self._qubrid and self._qubrid.is_available:
                    return self._qubrid.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
            except Exception as e:
                logger.warning("Inference source '%s' failed: %s", source, e)
                errors.append(f"{source}: {e}")
                continue

        raise RuntimeError(
            "All inference sources unavailable. "
            "Please check your connection or download a local model. "
            f"Errors: {'; '.join(errors) if errors else 'no sources configured'}"
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        model: Optional[str] = None,
        preferred_location: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> Iterator[str]:
        logger.debug(
            "InferenceRouter.stream — model hint=%r, preferred_location=%r, tier=%r",
            model,
            preferred_location,
            tier,
        )
        errors: list[str] = []
        for source in self._priority:
            try:
                if source == "local" and self._engine.is_loaded:
                    yield from self._engine.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
                if source == "lan" and self._lan and self._lan.is_available:
                    yield from self._lan.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
                if source == "qubrid" and self._qubrid and self._qubrid.is_available:
                    yield from self._qubrid.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
            except Exception as e:
                logger.warning("Inference source '%s' failed: %s", source, e)
                errors.append(f"{source}: {e}")
                continue

        raise RuntimeError(
            "All inference sources unavailable. "
            f"Errors: {'; '.join(errors) if errors else 'no sources configured'}"
        )
