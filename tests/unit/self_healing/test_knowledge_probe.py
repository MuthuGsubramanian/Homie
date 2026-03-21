# tests/unit/self_healing/test_knowledge_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.knowledge_probe import KnowledgeProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestKnowledgeProbe:
    def test_healthy_when_rag_works(self):
        rag = MagicMock()
        rag._file_hashes = {"a.py": "abc123"}
        rag._search = MagicMock()
        probe = KnowledgeProbe(rag_pipeline=rag)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_no_indexed_files(self):
        rag = MagicMock()
        rag._file_hashes = {}
        rag._search = MagicMock()
        probe = KnowledgeProbe(rag_pipeline=rag)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_search_raises(self):
        rag = MagicMock()
        rag._file_hashes = {"a.py": "abc"}
        rag._search = MagicMock()
        rag._search.search.side_effect = RuntimeError("index corrupt")
        probe = KnowledgeProbe(rag_pipeline=rag)
        result = probe.check()
        assert result.status == HealthStatus.FAILED

    def test_handles_none_rag(self):
        probe = KnowledgeProbe(rag_pipeline=None)
        result = probe.check()
        assert result.status == HealthStatus.UNKNOWN
