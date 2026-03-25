"""Tests for finetune configuration models."""

from homie_core.finetune.config import (
    DataConfig,
    EvaluationConfig,
    FinetuneConfig,
    LimitsConfig,
    ScheduleConfig,
    TrainingConfig,
)


class TestFinetuneConfig:
    def test_defaults(self):
        cfg = FinetuneConfig()
        assert cfg.enabled is True
        assert cfg.base_model == "lfm2"
        assert cfg.registry_name == "PyMasters/Homie"

    def test_schedule_defaults(self):
        cfg = ScheduleConfig()
        assert cfg.business_hours_start == 8
        assert cfg.business_hours_end == 18
        assert cfg.business_days == [0, 1, 2, 3, 4]
        assert cfg.min_idle_minutes == 30

    def test_training_defaults(self):
        cfg = TrainingConfig()
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.sft_learning_rate == 2e-4
        assert cfg.dpo_learning_rate == 5e-5
        assert cfg.batch_size == 4
        assert cfg.max_seq_length == 4096

    def test_data_defaults(self):
        cfg = DataConfig()
        assert cfg.sft_per_cycle == 2000
        assert cfg.dpo_per_cycle == 1000
        assert cfg.min_quality_score == 4
        assert cfg.include_curated is False

    def test_evaluation_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.promotion_threshold == 0.02
        assert cfg.safety_floor == 0.85
        assert cfg.plateau_cycles == 3

    def test_limits_defaults(self):
        cfg = LimitsConfig()
        assert cfg.max_cycles == 10
        assert cfg.max_disk_gb == 50
        assert cfg.max_lora_rank == 32

    def test_nested_in_homie_config(self):
        from homie_core.config import HomieConfig
        cfg = HomieConfig()
        assert hasattr(cfg, "finetune")
        assert cfg.finetune.enabled is True
