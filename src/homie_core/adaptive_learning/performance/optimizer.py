"""PerformanceOptimizer — coordinates caching, context optimization, and resource scheduling."""

import logging
from typing import Optional

from ..observation.signals import LearningSignal
from ..storage import LearningStorage
from .context_optimizer import ContextOptimizer
from .resource_scheduler import ResourceScheduler
from .response_cache import ResponseCache

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Coordinates response caching, context relevance, and resource scheduling."""

    def __init__(
        self,
        storage: LearningStorage,
        cache_max_entries: int = 500,
        cache_ttl: float = 86400.0,
        similarity_threshold: float = 0.92,
    ) -> None:
        self._storage = storage
        self.cache = ResponseCache(
            max_entries=cache_max_entries,
            ttl_default=cache_ttl,
            similarity_threshold=similarity_threshold,
        )
        self.context_optimizer = ContextOptimizer(storage=storage)
        self.resource_scheduler = ResourceScheduler()

    def on_signal(self, signal: LearningSignal) -> None:
        """Process performance-related signals."""
        data = signal.data
        if "hour" in data and "activity" in data:
            self.resource_scheduler.record_activity(data["hour"], data["activity"])

    def cache_response(self, query: str, response: str, context_hash: str = "") -> None:
        """Cache a query-response pair."""
        self.cache.put(query, response, context_hash)

    def get_cached_response(self, query: str, context_hash: Optional[str] = None) -> Optional[str]:
        """Get a cached response."""
        return self.cache.get(query, context_hash)

    def rank_context(self, query_type: str, sources: list[str]) -> list[str]:
        """Rank context sources by learned relevance."""
        return self.context_optimizer.rank_sources(query_type, sources)

    def cache_stats(self) -> dict:
        """Get cache performance statistics."""
        return self.cache.stats()
