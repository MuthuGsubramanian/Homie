"""Integration test: model evolution lifecycle."""
import pytest
from unittest.mock import MagicMock
from homie_core.model_evolution.evolution_engine import EvolutionEngine
from homie_core.model_evolution.modelfile_builder import ModelfileBuilder
from homie_core.model_evolution.data_curator import DataCurator
from homie_core.adaptive_learning.storage import LearningStorage


class TestModelEvolutionLifecycle:
    def test_modelfile_builds_from_preferences(self):
        builder = ModelfileBuilder(base_model="lfm2", user_name="Master")
        builder.set_preferences(verbosity="concise", depth="expert", format_pref="bullets")
        builder.set_knowledge(["Works on Homie AI", "Uses Python"])
        builder.set_customizations(["/standup: git + calendar"])
        builder.set_parameters(temperature=0.5, num_ctx=32768)
        content = builder.build()
        assert "FROM lfm2" in content
        assert "Master" in content
        assert "concise" in content.lower()
        assert "Homie AI" in content
        assert "standup" in content.lower()
        assert "PARAMETER temperature 0.5" in content

    def test_evolution_with_milestone_trigger(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        engine = EvolutionEngine(
            storage=storage,
            ollama_manager=MagicMock(create=MagicMock(return_value=True)),
            preference_engine=MagicMock(get_active_profile=MagicMock(return_value=MagicMock(verbosity=0.3, formality=0.5, technical_depth=0.8, format_preference="bullets"))),
            knowledge_query=MagicMock(),
            customization_manager=MagicMock(list_customizations=MagicMock(return_value=[])),
            profiler=MagicMock(get_profile=MagicMock(return_value=None)),
            inference_fn=MagicMock(return_value="Homie here! Python is great for coding."),
            base_model="lfm2",
            registry_name="MSG-88/Homie",
            modelfile_dir=str(tmp_path / "modelfiles"),
            min_facts=1,
            benchmark_threshold=0.3,
        )
        engine.record_new_fact()
        result = engine.evolve()
        assert result["status"] in ("promoted", "benchmark_passed", "created")

    def test_training_data_curation(self, tmp_path):
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        curator = DataCurator(storage=storage)
        curator.collect_sft("sys", "What is Python?", "A language.", 0.9)
        curator.collect_dpo("Explain X", "Good explanation", "Bad explanation")
        stats = curator.get_stats()
        assert stats["sft"] == 1
        assert stats["dpo"] == 1
        # Export
        sft_count = curator.export_sft(tmp_path / "sft.jsonl")
        assert sft_count == 1
