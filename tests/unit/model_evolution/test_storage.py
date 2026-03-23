# tests/unit/model_evolution/test_storage.py
import pytest
from homie_core.adaptive_learning.storage import LearningStorage


class TestModelEvolutionStorage:
    def test_save_and_get_model_version(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_model_version("v1", {"version_id": "v1", "base_model": "lfm2", "status": "active", "ollama_name": "test", "modelfile_hash": "x", "metrics": "{}", "changelog": ""})
        result = store.get_active_model_version()
        assert result is not None
        assert result["version_id"] == "v1"

    def test_update_model_version_status(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_model_version("v1", {"version_id": "v1", "base_model": "lfm2", "status": "active", "ollama_name": "t", "modelfile_hash": "x", "metrics": "{}", "changelog": ""})
        store.update_model_version_status("v1", "archived")
        result = store.get_active_model_version()
        assert result is None

    def test_save_and_get_training_example(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_training_example(example_type="sft", data='{"input": "hi"}', quality_score=0.9)
        examples = store.get_training_examples(example_type="sft")
        assert len(examples) == 1

    def test_count_training_examples(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_training_example(example_type="sft", data='{}', quality_score=0.8)
        store.save_training_example(example_type="dpo", data='{}', quality_score=0.0)
        counts = store.count_training_examples()
        assert counts["sft"] == 1
        assert counts["dpo"] == 1
