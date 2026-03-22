# tests/unit/knowledge_evolution/test_inference_engine.py
import pytest
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore
from homie_core.adaptive_learning.knowledge.reasoning.inference_engine import InferenceEngine


class TestInferenceEngine:
    def _setup(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        user = store.add_entity("User", "person")
        homie = store.add_entity("Homie", "project")
        python = store.add_entity("Python", "technology")
        store.add_relationship(user, "works_on", homie, confidence=0.95, source="conversation")
        store.add_relationship(homie, "uses", python, confidence=0.9, source="code_scan")
        return store, {"user": user, "homie": homie, "python": python}

    def test_infer_transitive(self, tmp_path):
        store, ids = self._setup(tmp_path)
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        # Should infer: User works_with Python
        assert len(inferred) >= 1
        assert any(r["relation"] == "works_with" for r in inferred)

    def test_inferred_has_lower_confidence(self, tmp_path):
        store, ids = self._setup(tmp_path)
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        for r in inferred:
            assert r["confidence"] < 0.95  # lower than source facts

    def test_inferred_marked_as_inference_source(self, tmp_path):
        store, ids = self._setup(tmp_path)
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        for r in inferred:
            assert r["source"] == "inference"

    def test_no_duplicate_inference(self, tmp_path):
        store, ids = self._setup(tmp_path)
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred1 = engine.run_inference()
        inferred2 = engine.run_inference()
        # Second run should not create duplicates
        all_rels = store.get_relationships(subject_id=ids["user"])
        inference_rels = [r for r in all_rels if r["source"] == "inference"]
        assert len(inference_rels) <= len(inferred1)

    def test_respects_max_hops(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        a = store.add_entity("A", "thing")
        b = store.add_entity("B", "thing")
        c = store.add_entity("C", "thing")
        d = store.add_entity("D", "thing")
        store.add_relationship(a, "uses", b, confidence=0.9, source="test")
        store.add_relationship(b, "uses", c, confidence=0.9, source="test")
        store.add_relationship(c, "uses", d, confidence=0.9, source="test")
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        # Should not infer A->D (3 hops)
        assert not any(r["subject_id"] == a and r["object_id"] == d for r in inferred)
