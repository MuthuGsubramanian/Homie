# tests/unit/knowledge_evolution/test_contradiction_detector.py
import time
import pytest
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore
from homie_core.adaptive_learning.knowledge.reasoning.contradiction_detector import ContradictionDetector


class TestContradictionDetector:
    def _setup(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        return store

    def test_detect_contradiction(self, tmp_path):
        store = self._setup(tmp_path)
        user = store.add_entity("User", "person")
        google = store.add_entity("Google", "organization")
        anthropic = store.add_entity("Anthropic", "organization")
        store.add_relationship(user, "works_at", google, confidence=0.8, source="conversation")
        store.add_relationship(user, "works_at", anthropic, confidence=0.9, source="conversation")
        detector = ContradictionDetector(graph_store=store)
        contradictions = detector.detect()
        assert len(contradictions) >= 1

    def test_no_contradiction_for_different_predicates(self, tmp_path):
        store = self._setup(tmp_path)
        user = store.add_entity("User", "person")
        google = store.add_entity("Google", "organization")
        store.add_relationship(user, "works_at", google, confidence=0.9, source="conv")
        store.add_relationship(user, "admires", google, confidence=0.7, source="conv")
        detector = ContradictionDetector(graph_store=store)
        contradictions = detector.detect()
        assert len(contradictions) == 0

    def test_resolve_by_confidence(self, tmp_path):
        store = self._setup(tmp_path)
        user = store.add_entity("User", "person")
        google = store.add_entity("Google", "organization")
        anthropic = store.add_entity("Anthropic", "organization")
        r1 = store.add_relationship(user, "works_at", google, confidence=0.7, source="conv")
        r2 = store.add_relationship(user, "works_at", anthropic, confidence=0.95, source="conv")
        detector = ContradictionDetector(graph_store=store)
        resolved = detector.resolve_all()
        assert len(resolved) >= 1
        # Lower confidence should be superseded
        google_rels = store.find_current_relationships(user, "works_at")
        assert len(google_rels) == 1
        assert google_rels[0]["object_id"] == anthropic

    def test_no_resolve_needed_when_clean(self, tmp_path):
        store = self._setup(tmp_path)
        user = store.add_entity("User", "person")
        google = store.add_entity("Google", "organization")
        store.add_relationship(user, "works_at", google, confidence=0.9, source="conv")
        detector = ContradictionDetector(graph_store=store)
        resolved = detector.resolve_all()
        assert len(resolved) == 0
