# tests/unit/model_evolution/test_validator.py
import pytest
from unittest.mock import MagicMock
from homie_core.model_evolution.validator import ModelValidator, BenchmarkResult


class TestBenchmarkResult:
    def test_passes_above_threshold(self):
        r = BenchmarkResult(scores={"style": 0.8, "knowledge": 0.9, "instructions": 0.7}, threshold=0.7)
        assert r.passed is True
        assert r.average_score > 0.7

    def test_fails_below_threshold(self):
        r = BenchmarkResult(scores={"style": 0.5, "knowledge": 0.4}, threshold=0.7)
        assert r.passed is False


class TestModelValidator:
    def test_run_benchmark(self):
        inference_fn = MagicMock(return_value="Python is a programming language.")
        validator = ModelValidator(inference_fn=inference_fn, benchmark_threshold=0.5)
        result = validator.run_benchmark()
        assert isinstance(result, BenchmarkResult)
        assert len(result.scores) > 0

    def test_benchmark_calls_inference(self):
        inference_fn = MagicMock(return_value="response")
        validator = ModelValidator(inference_fn=inference_fn)
        validator.run_benchmark()
        assert inference_fn.call_count >= 3  # at least a few benchmark prompts

    def test_shadow_test_result(self):
        validator = ModelValidator(inference_fn=MagicMock(return_value="ok"), shadow_min_queries=2)
        validator.record_shadow_result(old_score=0.6, new_score=0.8)
        validator.record_shadow_result(old_score=0.7, new_score=0.9)
        assert validator.shadow_test_passed(min_win_rate=0.5) is True

    def test_shadow_test_fails_when_old_better(self):
        validator = ModelValidator(inference_fn=MagicMock(return_value="ok"), shadow_min_queries=2)
        for _ in range(5):
            validator.record_shadow_result(old_score=0.9, new_score=0.3)
        assert validator.shadow_test_passed(min_win_rate=0.6) is False

    def test_shadow_test_needs_enough_samples(self):
        validator = ModelValidator(inference_fn=MagicMock(return_value="ok"), shadow_min_queries=10)
        validator.record_shadow_result(old_score=0.5, new_score=0.9)
        assert validator.shadow_test_passed() is False  # not enough samples
