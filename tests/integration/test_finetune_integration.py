"""Integration tests for the finetuning pipeline wiring."""

import inspect
from unittest.mock import MagicMock

from homie_core.model_evolution.evolution_engine import EvolutionEngine


class TestFinetuneIntegration:
    def test_evolution_engine_has_finetune_method(self):
        engine = EvolutionEngine(
            storage=MagicMock(),
            ollama_manager=MagicMock(),
            preference_engine=MagicMock(),
            knowledge_query=MagicMock(),
            customization_manager=MagicMock(),
            profiler=MagicMock(),
            inference_fn=MagicMock(),
        )
        assert hasattr(engine, "evolve_finetune")
        assert callable(engine.evolve_finetune)

    def test_finetune_config_loaded(self):
        from homie_core.config import load_config

        cfg = load_config()
        assert hasattr(cfg, "finetune")
        assert cfg.finetune.enabled is True
        assert cfg.finetune.base_model == "lfm2"
        assert cfg.finetune.registry_name == "PyMasters/Homie"

    def test_registry_name_updated(self):
        sig = inspect.signature(EvolutionEngine.__init__)
        default = sig.parameters["registry_name"].default
        assert default == "PyMasters/Homie"
