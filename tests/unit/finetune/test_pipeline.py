"""Tests for PipelineState and RecursiveFinetuneLoop."""

from __future__ import annotations

from unittest.mock import MagicMock

from homie_core.finetune.config import FinetuneConfig
from homie_core.finetune.pipeline import PipelineState, RecursiveFinetuneLoop
from homie_core.finetune.synthetic.templates import Domain


class TestPipelineState:
    """PipelineState persistence tests."""

    def test_initial_state(self, tmp_path):
        state = PipelineState(tmp_path)
        assert state.current_cycle == 0
        assert state.lora_rank == 16
        assert state.plateau_counter == 0
        assert state.cycle_scores == {}
        # All domains should have difficulty tier 1
        for domain in Domain:
            assert state.difficulty_tiers[domain.value] == 1

    def test_save_and_load(self, tmp_path):
        state = PipelineState(tmp_path)
        state.current_cycle = 3
        state.lora_rank = 32
        state.plateau_counter = 2
        state.record_score(0, 0.75)
        state.record_score(1, 0.78)
        state.record_score(2, 0.80)
        state.save()

        loaded = PipelineState.load(tmp_path)
        assert loaded.current_cycle == 3
        assert loaded.lora_rank == 32
        assert loaded.plateau_counter == 2
        assert loaded.cycle_scores == {0: 0.75, 1: 0.78, 2: 0.80}


class TestRecursiveFinetuneLoop:
    """RecursiveFinetuneLoop unit tests."""

    def _make_pipeline(self, tmp_path) -> RecursiveFinetuneLoop:
        cfg = FinetuneConfig()
        return RecursiveFinetuneLoop(
            config=cfg,
            inference_fn=MagicMock(return_value="response"),
            ollama_manager=MagicMock(),
            model_registry=MagicMock(),
            base_dir=tmp_path,
        )

    def test_init(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        assert pipe.state.current_cycle == 0
        assert pipe.get_status()["stage"] == "idle"

    def test_should_stop_at_max_cycles(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        pipe.state.current_cycle = pipe.config.limits.max_cycles
        assert pipe._should_stop() is True

    def test_should_stop_on_plateau(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        pipe.state.plateau_counter = pipe.config.evaluation.plateau_cycles
        pipe.state.lora_rank = pipe.config.limits.max_lora_rank
        assert pipe._should_stop() is True

    def test_should_not_stop_on_plateau_if_rank_can_escalate(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        pipe.state.plateau_counter = pipe.config.evaluation.plateau_cycles
        pipe.state.lora_rank = 16  # less than max_lora_rank (32)
        assert pipe._should_stop() is False

    def test_escalate_lora_on_plateau(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        pipe.state.lora_rank = 16
        pipe._handle_plateau()
        assert pipe.state.lora_rank == 32
        assert pipe.state.plateau_counter == 0

    def test_escalate_lora_capped_at_max(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        pipe.state.lora_rank = 32  # already at max
        pipe._handle_plateau()
        assert pipe.state.lora_rank == 32  # should not exceed max

    def test_get_difficulty(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        assert pipe._get_difficulty(0.95) == 4
        assert pipe._get_difficulty(0.85) == 3
        assert pipe._get_difficulty(0.70) == 2
        assert pipe._get_difficulty(0.50) == 1

    def test_load_accumulated_data_empty(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        data = pipe._load_accumulated_data("sft.jsonl")
        assert data == []

    def test_load_accumulated_data_with_files(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        pipe.state.current_cycle = 1
        # Create cycle-0 and cycle-1 dataset dirs
        import json
        for cycle in range(2):
            ds_dir = tmp_path / "datasets" / f"cycle-{cycle}"
            ds_dir.mkdir(parents=True, exist_ok=True)
            with open(ds_dir / "sft.jsonl", "w") as f:
                f.write(json.dumps({"cycle": cycle, "idx": 0}) + "\n")
                f.write(json.dumps({"cycle": cycle, "idx": 1}) + "\n")

        data = pipe._load_accumulated_data("sft.jsonl")
        assert len(data) == 4
        assert data[0]["cycle"] == 0
        assert data[2]["cycle"] == 1
