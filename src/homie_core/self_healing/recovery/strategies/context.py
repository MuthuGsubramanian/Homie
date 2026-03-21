"""Recovery strategies for context/observer failures."""

import logging
from ..engine import RecoveryResult, RecoveryTier

logger = logging.getLogger(__name__)


def restart_observer(module, status, error, context_aggregator=None, **ctx) -> RecoveryResult:
    """T1: Restart the context aggregator and verify it works."""
    if context_aggregator is None:
        return RecoveryResult(success=False, action="no context aggregator", tier=RecoveryTier.RETRY)
    try:
        snapshot = context_aggregator.tick()
        if snapshot:
            return RecoveryResult(success=True, action="observer restarted and responding", tier=RecoveryTier.RETRY)
        return RecoveryResult(success=False, action="observer returned empty after restart", tier=RecoveryTier.RETRY)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"observer restart failed: {exc}", tier=RecoveryTier.RETRY)


def reduce_monitoring_frequency(module, status, error, config=None, **ctx) -> RecoveryResult:
    """T2: Reduce observer polling frequency to reduce load."""
    logger.info("Reducing context monitoring frequency")
    return RecoveryResult(
        success=True,
        action="reduced monitoring frequency",
        tier=RecoveryTier.FALLBACK,
        details={"mode": "reduced_frequency"},
    )


def degrade_without_context(module, status, error, **ctx) -> RecoveryResult:
    """T4: Disable context observers, continue without context awareness."""
    logger.warning("Context observers disabled — running without context awareness")
    return RecoveryResult(
        success=True,
        action="context disabled, running without observers",
        tier=RecoveryTier.DEGRADE,
        details={"mode": "no_context"},
    )
