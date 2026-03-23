# tests/unit/model_evolution/test_evolution_engine.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.model_evolution.evolution_engine import EvolutionEngine


class TestEvolutionEngine:
    def _make_engine(self, **overrides):
        defaults = {
            "storage": MagicMock(),
            "ollama_manager": MagicMock(),
            "preference_engine": MagicMock(),
            "knowledge_query": MagicMock(),
            "customization_manager": MagicMock(),
            "profiler": MagicMock(),
            "inference_fn": MagicMock(return_value="test response"),
            "base_model": "lfm2",
            "registry_name": "MSG-88/Homie",
            "user_name": "Master",
            "modelfile_dir": "/tmp/homie",
        }
        defaults.update(overrides)
        return EvolutionEngine(**defaults)

    def test_build_modelfile(self):
        pref = MagicMock()
        pref.get_active_profile.return_value = MagicMock(verbosity=0.2, formality=0.3, technical_depth=0.8, format_preference="bullets")
        km = MagicMock()
        km.list_customizations.return_value = [{"request_text": "/standup command", "status": "active"}]
        engine = self._make_engine(preference_engine=pref, customization_manager=km)
        builder = engine.build_modelfile()
        content = builder.build()
        assert "FROM lfm2" in content
        assert "Master" in content

    def test_check_milestone_returns_false_initially(self):
        engine = self._make_engine()
        assert engine.should_evolve() is False

    def test_manual_trigger(self):
        engine = self._make_engine()
        engine.trigger_evolution()
        assert engine.should_evolve() is True

    def test_evolve_creates_model(self):
        ollama = MagicMock()
        ollama.create.return_value = True
        storage = MagicMock()
        storage.get_active_model_version.return_value = None
        storage.get_previous_model_version.return_value = None
        engine = self._make_engine(ollama_manager=ollama, storage=storage, benchmark_threshold=0.3)
        engine.trigger_evolution()
        result = engine.evolve()
        assert result["status"] in ("created", "benchmark_passed", "promoted")
        ollama.create.assert_called()

    def test_evolve_skips_if_no_milestone(self):
        engine = self._make_engine()
        result = engine.evolve()
        assert result["status"] == "no_changes"
