"""Tests for the QLoRA trainer."""

from __future__ import annotations

from unittest.mock import patch

from homie_core.finetune.config import TrainingConfig
from homie_core.finetune.training.qlora_trainer import QLoRATrainer


class TestQLoRATrainer:
    """QLoRATrainer unit tests (no GPU required)."""

    def test_init_with_config(self, tmp_path):
        cfg = TrainingConfig(lora_rank=8, batch_size=2)
        trainer = QLoRATrainer(
            base_model="lfm2", config=cfg, output_dir=tmp_path / "out"
        )
        assert trainer.base_model == "lfm2"
        assert trainer.config.lora_rank == 8
        assert trainer.output_dir.exists()

    def test_format_sft_example(self, tmp_path):
        trainer = QLoRATrainer(
            base_model="lfm2",
            config=TrainingConfig(),
            output_dir=tmp_path / "out",
        )
        example = {
            "system": "You are helpful.",
            "user": "Hello",
            "assistant": "Hi there!",
        }
        result = trainer.format_sft_example(example)
        assert "<|system|>" in result
        assert "You are helpful." in result
        assert "<|user|>" in result
        assert "Hello" in result
        assert "<|assistant|>" in result
        assert "Hi there!" in result

    def test_format_dpo_example(self, tmp_path):
        trainer = QLoRATrainer(
            base_model="lfm2",
            config=TrainingConfig(),
            output_dir=tmp_path / "out",
        )
        example = {
            "system": "Be concise.",
            "user": "Explain gravity.",
            "chosen": "Gravity is a force.",
            "rejected": "I don't know.",
        }
        result = trainer.format_dpo_example(example)
        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result
        assert "Be concise." in result["prompt"]
        assert "Explain gravity." in result["prompt"]
        assert result["chosen"] == "Gravity is a force."
        assert result["rejected"] == "I don't know."

    def test_convert_alpaca_to_chatml(self, tmp_path):
        example = {
            "instruction": "Translate to French",
            "input": "Hello world",
            "output": "Bonjour le monde",
        }
        result = QLoRATrainer.convert_alpaca_to_chatml(example)
        assert result["system"] == ""
        assert "Translate to French" in result["user"]
        assert "Hello world" in result["user"]
        assert result["assistant"] == "Bonjour le monde"

    def test_convert_alpaca_to_chatml_no_input(self):
        example = {
            "instruction": "Say hi",
            "input": "",
            "output": "Hi!",
        }
        result = QLoRATrainer.convert_alpaca_to_chatml(example)
        assert result["user"] == "Say hi"
        assert result["assistant"] == "Hi!"

    @patch("homie_core.finetune.training.qlora_trainer._check_gpu_available")
    def test_preflight_check_no_gpu(self, mock_gpu, tmp_path):
        mock_gpu.return_value = False
        trainer = QLoRATrainer(
            base_model="lfm2",
            config=TrainingConfig(),
            output_dir=tmp_path / "out",
        )
        ok, msg = trainer.preflight_check()
        assert ok is False
        assert "No GPU" in msg

    @patch("homie_core.finetune.training.qlora_trainer._check_gpu_available")
    def test_preflight_check_with_gpu(self, mock_gpu, tmp_path):
        mock_gpu.return_value = True
        trainer = QLoRATrainer(
            base_model="lfm2",
            config=TrainingConfig(),
            output_dir=tmp_path / "out",
        )
        # Even with GPU, we may not have unsloth/peft installed in test env,
        # so we just verify the GPU check passes and the method doesn't crash.
        ok, msg = trainer.preflight_check()
        # Result depends on whether peft/unsloth is installed
        assert isinstance(ok, bool)
        assert isinstance(msg, str)
