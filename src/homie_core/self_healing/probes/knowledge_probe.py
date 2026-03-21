"""Health probe for knowledge/RAG pipeline."""

from .base import BaseProbe, HealthStatus, ProbeResult


class KnowledgeProbe(BaseProbe):
    """Checks RAG pipeline, document index, and search health."""

    name = "knowledge"
    interval = 30.0

    def __init__(self, rag_pipeline=None) -> None:
        self._rag = rag_pipeline

    def check(self) -> ProbeResult:
        if self._rag is None:
            return ProbeResult(
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                error_count=0,
                last_error="RAG pipeline not initialized",
            )

        indexed_count = len(getattr(self._rag, "_file_hashes", {}))
        metadata = {"indexed_files": indexed_count}

        # Test search functionality if files are indexed
        if indexed_count > 0:
            try:
                self._rag._search.search("health_check", n=1)
            except Exception as exc:
                return ProbeResult(
                    status=HealthStatus.FAILED,
                    latency_ms=0,
                    error_count=1,
                    last_error=f"Search failed: {exc}",
                    metadata=metadata,
                )

        if indexed_count == 0:
            return ProbeResult(
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error_count=0,
                last_error="No files indexed",
                metadata=metadata,
            )

        return ProbeResult(
            status=HealthStatus.HEALTHY,
            latency_ms=0,
            error_count=0,
            metadata=metadata,
        )
