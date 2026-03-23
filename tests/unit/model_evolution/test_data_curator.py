import pytest
from unittest.mock import MagicMock
from homie_core.model_evolution.data_curator import DataCurator


class TestDataCurator:
    def test_collect_sft_example(self):
        storage = MagicMock()
        curator = DataCurator(storage=storage)
        curator.collect_sft(
            system_prompt="You are Homie.",
            user_message="What is Python?",
            response="A programming language.",
            quality_score=0.8,
        )
        storage.save_training_example.assert_called_once()

    def test_collect_dpo_pair(self):
        storage = MagicMock()
        curator = DataCurator(storage=storage)
        curator.collect_dpo(
            user_message="Explain decorators",
            chosen="Good response here",
            rejected="Bad response here",
        )
        storage.save_training_example.assert_called_once()
        call_data = storage.save_training_example.call_args[1]
        assert call_data["example_type"] == "dpo"

    def test_export_sft_jsonl(self, tmp_path):
        storage = MagicMock()
        storage.get_training_examples.return_value = [
            {"example_type": "sft", "data": '{"instruction": "sys", "input": "hi", "output": "hello"}', "quality_score": 0.9},
        ]
        curator = DataCurator(storage=storage)
        path = tmp_path / "sft.jsonl"
        count = curator.export_sft(path)
        assert count == 1
        assert path.exists()

    def test_export_dpo_jsonl(self, tmp_path):
        storage = MagicMock()
        storage.get_training_examples.return_value = [
            {"example_type": "dpo", "data": '{"prompt": "q", "chosen": "good", "rejected": "bad"}', "quality_score": 0.0},
        ]
        curator = DataCurator(storage=storage)
        path = tmp_path / "dpo.jsonl"
        count = curator.export_dpo(path)
        assert count == 1

    def test_get_stats(self):
        storage = MagicMock()
        storage.count_training_examples.return_value = {"sft": 100, "dpo": 25}
        curator = DataCurator(storage=storage)
        stats = curator.get_stats()
        assert stats["sft"] == 100
        assert stats["dpo"] == 25
