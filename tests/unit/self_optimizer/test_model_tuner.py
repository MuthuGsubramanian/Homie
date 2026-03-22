import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.performance.self_optimizer.model_tuner import ModelTuner


class TestModelTuner:
    def _make_tuner(self, profiler=None):
        profiler = profiler or MagicMock()
        profiler.get_profile.return_value = None
        return ModelTuner(profiler=profiler)

    def test_trivial_query_gets_low_tokens(self):
        tuner = self._make_tuner()
        params = tuner.select_parameters("trivial")
        assert params["max_tokens"] <= 256
        assert params["temperature"] >= 0.7

    def test_complex_query_gets_high_tokens(self):
        tuner = self._make_tuner()
        params = tuner.select_parameters("complex")
        assert params["max_tokens"] >= 1536

    def test_coding_query_gets_low_temperature(self):
        tuner = self._make_tuner()
        params = tuner.select_parameters("moderate", query_hint="coding")
        assert params["temperature"] <= 0.5

    def test_creative_query_gets_high_temperature(self):
        tuner = self._make_tuner()
        params = tuner.select_parameters("moderate", query_hint="creative")
        assert params["temperature"] >= 0.8

    def test_uses_profile_when_available(self):
        profiler = MagicMock()
        profile_mock = MagicMock()
        profile_mock.temperature = 0.4
        profile_mock.max_tokens = 800
        profile_mock.sample_count = 20
        profile_mock.avg_response_tokens = 0
        profiler.get_profile.return_value = profile_mock
        tuner = ModelTuner(profiler=profiler)
        params = tuner.select_parameters("moderate", query_hint="coding")
        assert params["temperature"] == 0.4
        assert params["max_tokens"] == 800

    def test_record_result(self):
        profiler = MagicMock()
        profiler.get_profile.return_value = None
        tuner = ModelTuner(profiler=profiler)
        tuner.record_result("coding", temperature=0.5, max_tokens=512, response_tokens=200, latency_ms=300)
        profiler.record_observation.assert_called_once()
