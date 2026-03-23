# tests/unit/model_evolution/test_config.py
import pytest
from homie_core.config import ModelEvolutionConfig


class TestModelEvolutionConfig:
    def test_defaults(self):
        cfg = ModelEvolutionConfig()
        assert cfg.enabled is True
        assert cfg.ollama_registry_name == "MSG-88/Homie"
        assert cfg.ollama_base_model == "lfm2"
        assert cfg.milestones_min_facts == 50
        assert cfg.validation_benchmark_min_score == 0.7
