"""Configuration models for the recursive finetuning pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScheduleConfig(BaseModel):
    """Controls when finetuning cycles are allowed to run."""

    business_hours_start: int = 8
    business_hours_end: int = 18
    business_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    min_idle_minutes: int = 30
    check_interval_minutes: int = 30


class TrainingConfig(BaseModel):
    """Hyperparameters for LoRA / SFT / DPO training."""

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    sft_learning_rate: float = 2e-4
    dpo_learning_rate: float = 5e-5
    sft_epochs: int = 3
    dpo_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_seq_length: int = 4096
    checkpoint_steps: int = 50
    warmup_ratio: float = 0.05


class DataConfig(BaseModel):
    """Controls synthetic data generation volumes and quality gates."""

    sft_per_cycle: int = 2000
    dpo_per_cycle: int = 1000
    min_quality_score: int = 4
    weak_domain_boost: float = 0.4
    include_curated: bool = False


class EvaluationConfig(BaseModel):
    """Thresholds for model promotion and regression checks."""

    promotion_threshold: float = 0.02
    max_regression_per_domain: float = 0.05
    safety_floor: float = 0.85
    plateau_cycles: int = 3


class LimitsConfig(BaseModel):
    """Safety limits to prevent runaway training."""

    max_cycles: int = 10
    max_disk_gb: int = 50
    max_lora_rank: int = 32


class FinetuneConfig(BaseModel):
    """Top-level configuration for the recursive finetuning pipeline."""

    enabled: bool = True
    base_model: str = "lfm2"
    registry_name: str = "PyMasters/Homie"
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
