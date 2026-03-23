"""Model validator — benchmark suite and A/B shadow testing."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Fixed benchmark prompts
_BENCHMARK_PROMPTS = [
    {"category": "style", "prompt": "Explain what a Python decorator is.", "expected_traits": ["concise", "code"]},
    {"category": "knowledge", "prompt": "What projects am I working on?", "expected_traits": ["project", "working"]},
    {"category": "instructions", "prompt": "List three benefits of testing.", "expected_traits": ["1", "2", "3"]},
    {"category": "personality", "prompt": "Good morning!", "expected_traits": ["homie", "morning", "hello", "hey"]},
    {"category": "format", "prompt": "Give me a bullet point summary of Python features.", "expected_traits": ["-", "*", "•"]},
]


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    scores: dict[str, float]
    threshold: float = 0.7

    @property
    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    @property
    def passed(self) -> bool:
        return self.average_score >= self.threshold


class ModelValidator:
    """Validates model quality via benchmarks and shadow testing."""

    def __init__(
        self,
        inference_fn: Callable[[str], str],
        benchmark_threshold: float = 0.7,
        shadow_min_queries: int = 50,
    ) -> None:
        self._infer = inference_fn
        self._benchmark_threshold = benchmark_threshold
        self._shadow_min = shadow_min_queries
        self._shadow_results: list[tuple[float, float]] = []  # (old_score, new_score)

    def run_benchmark(self) -> BenchmarkResult:
        """Run the benchmark suite against the model."""
        scores = {}
        for bench in _BENCHMARK_PROMPTS:
            try:
                response = self._infer(bench["prompt"])
                score = self._score_response(response, bench["expected_traits"])
                scores[bench["category"]] = score
            except Exception as exc:
                logger.warning("Benchmark failed for %s: %s", bench["category"], exc)
                scores[bench["category"]] = 0.0

        result = BenchmarkResult(scores=scores, threshold=self._benchmark_threshold)
        logger.info("Benchmark: avg=%.2f passed=%s", result.average_score, result.passed)
        return result

    def _score_response(self, response: str, expected_traits: list[str]) -> float:
        """Score a response based on expected traits."""
        if not response:
            return 0.0
        response_lower = response.lower()
        matches = sum(1 for t in expected_traits if t.lower() in response_lower)
        # Base score: 0.5 for any response, up to 1.0 for matching traits
        trait_score = matches / max(len(expected_traits), 1)
        return min(1.0, 0.5 + 0.5 * trait_score)

    def record_shadow_result(self, old_score: float, new_score: float) -> None:
        """Record a shadow test comparison."""
        self._shadow_results.append((old_score, new_score))

    def shadow_test_passed(self, min_win_rate: float = 0.6) -> bool:
        """Check if shadow testing has passed."""
        if len(self._shadow_results) < self._shadow_min:
            return False
        wins = sum(1 for old, new in self._shadow_results if new >= old)
        win_rate = wins / len(self._shadow_results)
        return win_rate >= min_win_rate

    def reset_shadow(self) -> None:
        """Reset shadow test results."""
        self._shadow_results.clear()
