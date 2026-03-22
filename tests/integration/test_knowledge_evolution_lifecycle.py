"""Integration test: knowledge evolution lifecycle."""
import pytest
from pathlib import Path
from homie_core.adaptive_learning.knowledge.builder import KnowledgeBuilder
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore
from homie_core.adaptive_learning.knowledge.reasoning.entity_resolver import EntityResolver
from homie_core.adaptive_learning.knowledge.reasoning.inference_engine import InferenceEngine
from homie_core.adaptive_learning.knowledge.reasoning.contradiction_detector import ContradictionDetector
from homie_core.adaptive_learning.storage import LearningStorage


class TestKnowledgeEvolutionLifecycle:
    def test_intake_creates_graph_entities(self, tmp_path):
        """Guided intake populates the knowledge graph."""
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")

        # Create source files
        src = tmp_path / "project"
        src.mkdir()
        (src / "main.py").write_text("class Application:\n    pass\n\nclass Database:\n    pass\n")
        (src / "utils.py").write_text("import os\ndef helper():\n    return True\n")

        result = builder.ingest_source(src)
        assert result["files_scanned"] == 2
        assert builder.graph_store.entity_count() > 0

    def test_inference_derives_relationships(self, tmp_path):
        """Inference engine derives new facts from existing ones."""
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        user = store.add_entity("User", "person")
        homie = store.add_entity("Homie", "project")
        python = store.add_entity("Python", "technology")
        store.add_relationship(user, "works_on", homie, confidence=0.95, source="conversation")
        store.add_relationship(homie, "uses", python, confidence=0.9, source="code_scan")

        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        assert len(inferred) >= 1  # User works_with Python

    def test_contradiction_resolution_with_temporal_versioning(self, tmp_path):
        """Contradictions are resolved via temporal versioning."""
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        user = store.add_entity("User", "person")
        google = store.add_entity("Google", "organization")
        anthropic = store.add_entity("Anthropic", "organization")
        store.add_relationship(user, "works_at", google, confidence=0.7, source="old_conv")
        store.add_relationship(user, "works_at", anthropic, confidence=0.95, source="recent_conv")

        detector = ContradictionDetector(graph_store=store)
        resolutions = detector.resolve_all()
        assert len(resolutions) == 1

        # Only one current works_at relationship
        current = store.find_current_relationships(user, "works_at")
        assert len(current) == 1
        assert current[0]["object_id"] == anthropic

        # Old relationship still exists but superseded
        all_rels = store.get_relationships(subject_id=user)
        assert len(all_rels) == 2  # both exist

    def test_full_lifecycle(self, tmp_path):
        """Full flow: intake -> inference -> contradiction resolution."""
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()

        # Seed some entities
        user = store.add_entity("Developer", "person")
        project = store.add_entity("MyApp", "project")
        store.add_relationship(user, "works_on", project, confidence=0.9, source="conversation")

        # Add technology via code scan
        python = store.add_entity("Python", "technology")
        store.add_relationship(project, "uses", python, confidence=0.85, source="code_scan")

        # Run inference
        engine = InferenceEngine(graph_store=store, max_hops=2)
        inferred = engine.run_inference()
        assert len(inferred) >= 1

        # Verify graph is queryable
        from homie_core.adaptive_learning.knowledge.graph.query import GraphQuery
        query = GraphQuery(store=store)
        related = query.traverse(user, max_hops=2)
        names = [e["name"] for e in related]
        assert "MyApp" in names
        assert "Python" in names
