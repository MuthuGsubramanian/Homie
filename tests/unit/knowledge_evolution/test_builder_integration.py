# tests/unit/knowledge_evolution/test_builder_integration.py
import pytest
from homie_core.adaptive_learning.knowledge.builder import KnowledgeBuilder
from homie_core.adaptive_learning.storage import LearningStorage


class TestKnowledgeBuilderGraphIntegration:
    def test_has_graph_store(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")
        assert builder.graph_store is not None

    def test_has_intake_pipeline(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")
        assert builder.intake is not None

    def test_has_graph_query(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")
        assert builder.graph_query is not None

    def test_ingest_source(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")
        # Create a test source
        src_dir = tmp_path / "project"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("class App:\n    pass\n")
        result = builder.ingest_source(src_dir)
        assert result["files_scanned"] >= 1

    def test_process_turn_still_works(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        builder = KnowledgeBuilder(storage=storage, graph_db_path=tmp_path / "kg.db")
        facts = builder.process_turn("I work at Google", "Nice!")
        assert len(facts) >= 1
