# Performance Self-Optimizer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggressively optimize Homie's runtime by actively tuning prompts, model parameters, and pipeline stages — with hardware-aware persistent profiles.

**Architecture:** SelfOptimizer coordinator with three engines: PromptOptimizer (middleware that compresses context/history), ModelTuner (selects per-call temperature/max_tokens and manages load-time gpu_layers/context_length), PipelineGate (overrides complexity tier via CLASSIFIED hook to skip unnecessary stages). OptimizationProfiler persists learned parameters per (query_type, hardware_fingerprint).

**Tech Stack:** Python 3.11+, existing middleware hooks, existing CognitiveArchitecture complexity tiers, existing hardware detector.

**Spec:** `docs/superpowers/specs/2026-03-22-performance-self-optimizer-design.md`

---

## Chunk 1: Optimization Profiler & Model Tuner

### Task 1: Optimization Profile Model

**Files:**
- Create: `src/homie_core/adaptive_learning/performance/self_optimizer/__init__.py`
- Create: `src/homie_core/adaptive_learning/performance/self_optimizer/profiler.py`
- Test: `tests/unit/self_optimizer/__init__.py`
- Test: `tests/unit/self_optimizer/test_profiler.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/homie_core/adaptive_learning/performance/self_optimizer
mkdir -p tests/unit/self_optimizer
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/self_optimizer/__init__.py
```

```python
# tests/unit/self_optimizer/test_profiler.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.performance.self_optimizer.profiler import (
    OptimizationProfiler,
    OptimizationProfile,
    generate_hardware_fingerprint,
)


class TestOptimizationProfile:
    def test_defaults(self):
        p = OptimizationProfile(query_type="coding", hardware_fingerprint="abc123")
        assert p.temperature == 0.7
        assert p.max_tokens == 1024
        assert p.sample_count == 0

    def test_to_dict_from_dict(self):
        p = OptimizationProfile(query_type="chat", hardware_fingerprint="xyz", temperature=0.3, max_tokens=256)
        d = p.to_dict()
        p2 = OptimizationProfile.from_dict(d)
        assert p2.temperature == 0.3
        assert p2.max_tokens == 256

    def test_update_with_ema(self):
        p = OptimizationProfile(query_type="coding", hardware_fingerprint="abc", avg_response_tokens=100, sample_count=10)
        p.update(response_tokens=200, latency_ms=500, learning_rate=0.2)
        assert p.avg_response_tokens > 100  # moved toward 200
        assert p.avg_latency_ms > 0
        assert p.sample_count == 11


class TestHardwareFingerprint:
    def test_generates_string(self):
        fp = generate_hardware_fingerprint(gpu_name="RTX 4090", vram_mb=24576, ram_gb=32.0)
        assert isinstance(fp, str)
        assert len(fp) > 0

    def test_same_hardware_same_fingerprint(self):
        fp1 = generate_hardware_fingerprint(gpu_name="RTX 4090", vram_mb=24576, ram_gb=32.0)
        fp2 = generate_hardware_fingerprint(gpu_name="RTX 4090", vram_mb=24576, ram_gb=32.0)
        assert fp1 == fp2

    def test_different_hardware_different_fingerprint(self):
        fp1 = generate_hardware_fingerprint(gpu_name="RTX 4090", vram_mb=24576, ram_gb=32.0)
        fp2 = generate_hardware_fingerprint(gpu_name="RTX 3060", vram_mb=12288, ram_gb=16.0)
        assert fp1 != fp2


class TestOptimizationProfiler:
    def test_get_profile_returns_none_for_unknown(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        profiler = OptimizationProfiler(storage=storage, hardware_fingerprint="abc")
        assert profiler.get_profile("unknown_type") is None

    def test_save_and_get_profile(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        profiler = OptimizationProfiler(storage=storage, hardware_fingerprint="abc")
        profile = OptimizationProfile(query_type="coding", hardware_fingerprint="abc", temperature=0.4)
        profiler.save_profile(profile)
        storage.save_optimization_profile.assert_called_once()

    def test_record_observation_creates_profile(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        profiler = OptimizationProfiler(storage=storage, hardware_fingerprint="abc")
        profiler.record_observation("coding", temperature_used=0.5, max_tokens_used=512, response_tokens=200, latency_ms=300)
        storage.save_optimization_profile.assert_called()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_optimizer/test_profiler.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write implementation**

```python
# src/homie_core/adaptive_learning/performance/self_optimizer/__init__.py
"""Performance Self-Optimizer — active runtime tuning."""
```

```python
# src/homie_core/adaptive_learning/performance/self_optimizer/profiler.py
"""Optimization profiler — persists learned parameters per (query_type, hardware)."""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


def generate_hardware_fingerprint(gpu_name: str = "", vram_mb: int = 0, ram_gb: float = 0.0) -> str:
    """Generate a stable fingerprint for the current hardware."""
    raw = f"{gpu_name}:{vram_mb}:{ram_gb:.1f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class OptimizationProfile:
    """Learned optimization parameters for a query type on specific hardware."""

    query_type: str
    hardware_fingerprint: str
    temperature: float = 0.7
    max_tokens: int = 1024
    context_budget: int = 5000
    pipeline_tier: str = "moderate"
    avg_response_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    sample_count: int = 0

    def update(self, response_tokens: float, latency_ms: float, learning_rate: float = 0.1) -> None:
        """Update profile with new observation via EMA."""
        if self.sample_count == 0:
            self.avg_response_tokens = response_tokens
            self.avg_latency_ms = latency_ms
        else:
            self.avg_response_tokens = learning_rate * response_tokens + (1 - learning_rate) * self.avg_response_tokens
            self.avg_latency_ms = learning_rate * latency_ms + (1 - learning_rate) * self.avg_latency_ms
        self.sample_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "hardware_fingerprint": self.hardware_fingerprint,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_budget": self.context_budget,
            "pipeline_tier": self.pipeline_tier,
            "avg_response_tokens": self.avg_response_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class OptimizationProfiler:
    """Manages optimization profiles — save, load, update."""

    def __init__(self, storage, hardware_fingerprint: str) -> None:
        self._storage = storage
        self._hw_fp = hardware_fingerprint
        self._cache: dict[str, OptimizationProfile] = {}

    def get_profile(self, query_type: str) -> Optional[OptimizationProfile]:
        """Get profile for a query type on current hardware."""
        if query_type in self._cache:
            return self._cache[query_type]
        data = self._storage.get_optimization_profile(query_type, self._hw_fp)
        if data:
            profile = OptimizationProfile.from_dict(data)
            self._cache[query_type] = profile
            return profile
        return None

    def save_profile(self, profile: OptimizationProfile) -> None:
        """Save a profile to storage."""
        self._cache[profile.query_type] = profile
        self._storage.save_optimization_profile(profile.query_type, self._hw_fp, profile.to_dict())

    def record_observation(
        self,
        query_type: str,
        temperature_used: float,
        max_tokens_used: int,
        response_tokens: float,
        latency_ms: float,
    ) -> None:
        """Record an inference observation and update the profile."""
        profile = self.get_profile(query_type)
        if profile is None:
            profile = OptimizationProfile(
                query_type=query_type,
                hardware_fingerprint=self._hw_fp,
                temperature=temperature_used,
                max_tokens=max_tokens_used,
            )
        profile.update(response_tokens, latency_ms)
        self.save_profile(profile)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_optimizer/test_profiler.py -v`
Expected: All 9 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/adaptive_learning/performance/self_optimizer/ tests/unit/self_optimizer/
git commit -m "feat(self-optimizer): add OptimizationProfiler with hardware fingerprinting"
```

---

### Task 2: Model Tuner

**Files:**
- Create: `src/homie_core/adaptive_learning/performance/self_optimizer/model_tuner.py`
- Test: `tests/unit/self_optimizer/test_model_tuner.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_optimizer/test_model_tuner.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_optimizer/test_model_tuner.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/performance/self_optimizer/model_tuner.py
"""Model tuner — selects optimal per-call parameters based on query type and learned profiles."""

from typing import Any, Optional

from .profiler import OptimizationProfiler

# Default parameters by complexity tier
_TIER_DEFAULTS = {
    "trivial":  {"max_tokens": 256,  "temperature": 0.8},
    "simple":   {"max_tokens": 384,  "temperature": 0.7},
    "moderate": {"max_tokens": 768,  "temperature": 0.7},
    "complex":  {"max_tokens": 1536, "temperature": 0.6},
    "deep":     {"max_tokens": 3072, "temperature": 0.5},
}

# Query hint overrides for temperature
_HINT_TEMPERATURE = {
    "coding": 0.3,
    "factual": 0.3,
    "creative": 0.9,
    "chat": 0.7,
}

_MIN_PROFILE_SAMPLES = 5  # need at least this many observations before trusting profile


class ModelTuner:
    """Selects optimal inference parameters per query."""

    def __init__(self, profiler: OptimizationProfiler) -> None:
        self._profiler = profiler

    def select_parameters(
        self,
        complexity: str,
        query_hint: Optional[str] = None,
    ) -> dict[str, Any]:
        """Select optimal temperature and max_tokens for this query."""
        # Start with tier defaults
        defaults = _TIER_DEFAULTS.get(complexity, _TIER_DEFAULTS["moderate"])
        temperature = defaults["temperature"]
        max_tokens = defaults["max_tokens"]

        # Apply query hint
        if query_hint and query_hint in _HINT_TEMPERATURE:
            temperature = _HINT_TEMPERATURE[query_hint]

        # Check for learned profile
        profile_key = query_hint or complexity
        profile = self._profiler.get_profile(profile_key)
        if profile and profile.sample_count >= _MIN_PROFILE_SAMPLES:
            temperature = profile.temperature
            max_tokens = profile.max_tokens
            # Adjust max_tokens based on learned response length
            if profile.avg_response_tokens > 0:
                max_tokens = max(64, int(profile.avg_response_tokens * 1.5))

        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def record_result(
        self,
        query_type: str,
        temperature: float,
        max_tokens: int,
        response_tokens: float,
        latency_ms: float,
    ) -> None:
        """Record an inference result to update the profile."""
        self._profiler.record_observation(
            query_type=query_type,
            temperature_used=temperature,
            max_tokens_used=max_tokens,
            response_tokens=response_tokens,
            latency_ms=latency_ms,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_optimizer/test_model_tuner.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/performance/self_optimizer/model_tuner.py tests/unit/self_optimizer/test_model_tuner.py
git commit -m "feat(self-optimizer): add ModelTuner with per-call parameter selection"
```

---

## Chunk 2: Prompt Optimizer & Pipeline Gate

### Task 3: Prompt Optimizer

**Files:**
- Create: `src/homie_core/adaptive_learning/performance/self_optimizer/prompt_optimizer.py`
- Test: `tests/unit/self_optimizer/test_prompt_optimizer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_optimizer/test_prompt_optimizer.py
import pytest
from homie_core.adaptive_learning.performance.self_optimizer.prompt_optimizer import PromptOptimizer


class TestPromptOptimizer:
    def test_truncates_history_by_complexity(self):
        opt = PromptOptimizer()
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        trimmed = opt.trim_history(history, complexity="simple")
        assert len(trimmed) <= 5

    def test_preserves_all_history_for_deep(self):
        opt = PromptOptimizer()
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        trimmed = opt.trim_history(history, complexity="deep")
        assert len(trimmed) == 20

    def test_dedup_removes_similar_facts(self):
        opt = PromptOptimizer()
        facts = [
            "User works at Google",
            "The user is employed at Google",
            "Python is a programming language",
        ]
        deduped = opt.deduplicate_facts(facts)
        assert len(deduped) < len(facts)

    def test_dedup_preserves_unique_facts(self):
        opt = PromptOptimizer()
        facts = [
            "User works at Google",
            "Python is a programming language",
            "Homie uses ChromaDB",
        ]
        deduped = opt.deduplicate_facts(facts)
        assert len(deduped) == 3

    def test_compress_prompt_reduces_length(self):
        opt = PromptOptimizer()
        long_prompt = "You are Homie.\n" + "\n".join([f"Fact {i}: user likes thing {i}" for i in range(50)])
        compressed = opt.compress(long_prompt, complexity="simple", max_chars=500)
        assert len(compressed) <= len(long_prompt)

    def test_compress_respects_complexity_budget(self):
        opt = PromptOptimizer()
        prompt = "x" * 10000
        compressed_simple = opt.compress(prompt, complexity="simple", max_chars=1500)
        compressed_deep = opt.compress(prompt, complexity="deep", max_chars=12000)
        assert len(compressed_simple) <= 1500
        assert len(compressed_deep) <= 12000

    def test_middleware_interface(self):
        opt = PromptOptimizer()
        assert hasattr(opt, "modify_prompt")
        assert hasattr(opt, "name")
        assert opt.name == "prompt_optimizer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_optimizer/test_prompt_optimizer.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/performance/self_optimizer/prompt_optimizer.py
"""Prompt optimizer — compresses context and history to reduce token waste."""

from typing import Optional

from homie_core.middleware.base import HomieMiddleware

# Character budget by complexity (approximate — 4 chars ≈ 1 token)
_CHAR_BUDGETS = {
    "trivial":  1500,
    "simple":   2500,
    "moderate": 5000,
    "complex":  8000,
    "deep":     12000,
}

_HISTORY_LIMITS = {
    "trivial":  2,
    "simple":   5,
    "moderate": 10,
    "complex":  20,
    "deep":     999,  # effectively unlimited
}


class PromptOptimizer(HomieMiddleware):
    """Middleware that compresses prompts based on query complexity."""

    name = "prompt_optimizer"
    order = 50  # Run early — before other middleware adds more

    def __init__(self) -> None:
        self._current_complexity = "moderate"

    def set_complexity(self, complexity: str) -> None:
        """Set current query complexity for next prompt optimization."""
        self._current_complexity = complexity

    def modify_prompt(self, prompt: str) -> str:
        """Compress the prompt based on current complexity."""
        budget = _CHAR_BUDGETS.get(self._current_complexity, 5000)
        return self.compress(prompt, self._current_complexity, max_chars=budget)

    def compress(self, prompt: str, complexity: str, max_chars: int = 5000) -> str:
        """Compress a prompt to fit within a character budget."""
        if len(prompt) <= max_chars:
            return prompt
        # Strategy: keep the first part (system prompt) and last part (recent context)
        # Trim the middle (older context, less relevant facts)
        half = max_chars // 2
        return prompt[:half] + "\n...(compressed)...\n" + prompt[-half:]

    def trim_history(self, history: list[dict], complexity: str) -> list[dict]:
        """Trim conversation history based on complexity budget."""
        limit = _HISTORY_LIMITS.get(complexity, 10)
        if len(history) <= limit:
            return history
        # Keep most recent turns
        return history[-limit:]

    def deduplicate_facts(self, facts: list[str]) -> list[str]:
        """Remove semantically similar facts using simple word overlap."""
        if len(facts) <= 1:
            return facts

        unique = [facts[0]]
        for fact in facts[1:]:
            is_dup = False
            fact_words = set(fact.lower().split())
            for existing in unique:
                existing_words = set(existing.lower().split())
                if not fact_words or not existing_words:
                    continue
                overlap = len(fact_words & existing_words) / len(fact_words | existing_words)
                if overlap > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(fact)
        return unique
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_optimizer/test_prompt_optimizer.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/performance/self_optimizer/prompt_optimizer.py tests/unit/self_optimizer/test_prompt_optimizer.py
git commit -m "feat(self-optimizer): add PromptOptimizer middleware with compression and dedup"
```

---

### Task 4: Pipeline Gate

**Files:**
- Create: `src/homie_core/adaptive_learning/performance/self_optimizer/pipeline_gate.py`
- Test: `tests/unit/self_optimizer/test_pipeline_gate.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_optimizer/test_pipeline_gate.py
import pytest
from homie_core.adaptive_learning.performance.self_optimizer.pipeline_gate import PipelineGate


class TestPipelineGate:
    def test_trivial_overrides_to_trivial(self):
        gate = PipelineGate()
        result = gate.apply("trivial")
        assert result == "trivial"

    def test_complex_stays_complex(self):
        gate = PipelineGate()
        result = gate.apply("complex")
        assert result == "complex"

    def test_promotes_after_clarifications(self):
        gate = PipelineGate(promotion_threshold=2)
        gate.record_clarification("trivial")
        gate.record_clarification("trivial")
        # After 2 clarifications, trivial should be promoted
        result = gate.apply("trivial")
        assert result == "simple"  # promoted one tier

    def test_no_promote_below_threshold(self):
        gate = PipelineGate(promotion_threshold=3)
        gate.record_clarification("trivial")
        result = gate.apply("trivial")
        assert result == "trivial"  # not enough clarifications yet

    def test_double_promotion(self):
        gate = PipelineGate(promotion_threshold=2)
        # Promote trivial → simple
        gate.record_clarification("trivial")
        gate.record_clarification("trivial")
        # Then promote simple → moderate
        gate.record_clarification("simple")
        gate.record_clarification("simple")
        result = gate.apply("simple")
        assert result == "moderate"

    def test_deep_never_promotes(self):
        gate = PipelineGate(promotion_threshold=1)
        gate.record_clarification("deep")
        result = gate.apply("deep")
        assert result == "deep"  # can't promote beyond deep

    def test_get_skip_report(self):
        gate = PipelineGate()
        gate.apply("trivial")
        report = gate.get_stats()
        assert "trivial" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_optimizer/test_pipeline_gate.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/performance/self_optimizer/pipeline_gate.py
"""Pipeline gate — overrides complexity tier to skip unnecessary cognitive stages."""

import threading
from collections import defaultdict
from typing import Optional

# Tier promotion order
_TIER_ORDER = ["trivial", "simple", "moderate", "complex", "deep"]


class PipelineGate:
    """Overrides query complexity tier based on learned quality feedback."""

    def __init__(self, promotion_threshold: int = 3) -> None:
        self._threshold = promotion_threshold
        self._lock = threading.Lock()
        # {original_tier: promoted_tier}
        self._promotions: dict[str, str] = {}
        # {tier: clarification_count}
        self._clarification_counts: dict[str, int] = defaultdict(int)
        # Stats
        self._apply_counts: dict[str, int] = defaultdict(int)

    def apply(self, complexity: str) -> str:
        """Apply gating — returns the effective complexity tier."""
        with self._lock:
            self._apply_counts[complexity] += 1
            # Check if this tier has been promoted
            effective = self._promotions.get(complexity, complexity)
            return effective

    def record_clarification(self, tier: str) -> None:
        """Record that a gated query at this tier needed clarification."""
        with self._lock:
            self._clarification_counts[tier] += 1
            if self._clarification_counts[tier] >= self._threshold:
                # Promote to next tier
                idx = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else -1
                if idx >= 0 and idx < len(_TIER_ORDER) - 1:
                    promoted = _TIER_ORDER[idx + 1]
                    self._promotions[tier] = promoted
                    self._clarification_counts[tier] = 0  # reset

    def get_stats(self) -> dict[str, dict]:
        """Get gating statistics."""
        with self._lock:
            return {
                tier: {
                    "apply_count": self._apply_counts.get(tier, 0),
                    "clarifications": self._clarification_counts.get(tier, 0),
                    "promoted_to": self._promotions.get(tier),
                }
                for tier in _TIER_ORDER
                if self._apply_counts.get(tier, 0) > 0 or tier in self._promotions
            }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_optimizer/test_pipeline_gate.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/performance/self_optimizer/pipeline_gate.py tests/unit/self_optimizer/test_pipeline_gate.py
git commit -m "feat(self-optimizer): add PipelineGate with self-correcting tier promotion"
```

---

## Chunk 3: Coordinator, Config & Integration

### Task 5: SelfOptimizer Coordinator

**Files:**
- Create: `src/homie_core/adaptive_learning/performance/self_optimizer/coordinator.py`
- Test: `tests/unit/self_optimizer/test_coordinator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_optimizer/test_coordinator.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.performance.self_optimizer.coordinator import SelfOptimizer


class TestSelfOptimizer:
    def test_initializes_all_components(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc123")
        assert opt.prompt_optimizer is not None
        assert opt.model_tuner is not None
        assert opt.pipeline_gate is not None
        assert opt.profiler is not None

    def test_optimize_query_returns_params(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc")
        result = opt.optimize_query(complexity="moderate", query_hint="coding")
        assert "temperature" in result
        assert "max_tokens" in result
        assert "effective_complexity" in result

    def test_trivial_query_gets_fast_params(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc")
        result = opt.optimize_query(complexity="trivial")
        assert result["max_tokens"] <= 256
        assert result["effective_complexity"] == "trivial"

    def test_record_result_updates_profiler(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc")
        opt.record_result(
            query_type="coding",
            complexity="moderate",
            temperature=0.5,
            max_tokens=512,
            response_tokens=200,
            latency_ms=300,
        )
        storage.save_optimization_profile.assert_called()

    def test_record_clarification_feeds_gate(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc", promotion_threshold=1)
        opt.record_clarification("trivial")
        result = opt.optimize_query(complexity="trivial")
        assert result["effective_complexity"] == "simple"  # promoted

    def test_get_stats(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc")
        opt.optimize_query(complexity="moderate")
        stats = opt.get_stats()
        assert "gate" in stats
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_optimizer/test_coordinator.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/homie_core/adaptive_learning/performance/self_optimizer/coordinator.py
"""SelfOptimizer — coordinates prompt optimization, model tuning, and pipeline gating."""

import logging
from typing import Any, Optional

from .model_tuner import ModelTuner
from .pipeline_gate import PipelineGate
from .profiler import OptimizationProfiler
from .prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


class SelfOptimizer:
    """Coordinates all performance self-optimization."""

    def __init__(
        self,
        storage,
        hardware_fingerprint: str,
        promotion_threshold: int = 3,
    ) -> None:
        self.profiler = OptimizationProfiler(storage=storage, hardware_fingerprint=hardware_fingerprint)
        self.prompt_optimizer = PromptOptimizer()
        self.model_tuner = ModelTuner(profiler=self.profiler)
        self.pipeline_gate = PipelineGate(promotion_threshold=promotion_threshold)

    def optimize_query(
        self,
        complexity: str,
        query_hint: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get optimized parameters for a query."""
        # Apply pipeline gating
        effective = self.pipeline_gate.apply(complexity)

        # Set prompt optimizer's complexity
        self.prompt_optimizer.set_complexity(effective)

        # Get model parameters
        params = self.model_tuner.select_parameters(effective, query_hint=query_hint)

        return {
            "effective_complexity": effective,
            "original_complexity": complexity,
            **params,
        }

    def record_result(
        self,
        query_type: str,
        complexity: str,
        temperature: float,
        max_tokens: int,
        response_tokens: float,
        latency_ms: float,
    ) -> None:
        """Record an inference result for profile learning."""
        self.model_tuner.record_result(
            query_type=query_type,
            temperature=temperature,
            max_tokens=max_tokens,
            response_tokens=response_tokens,
            latency_ms=latency_ms,
        )

    def record_clarification(self, tier: str) -> None:
        """Record that a gated query needed clarification — may promote tier."""
        self.pipeline_gate.record_clarification(tier)

    def get_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        return {
            "gate": self.pipeline_gate.get_stats(),
        }
```

- [ ] **Step 4: Update self_optimizer __init__.py**

```python
# src/homie_core/adaptive_learning/performance/self_optimizer/__init__.py
"""Performance Self-Optimizer — active runtime tuning."""
from .coordinator import SelfOptimizer
from .model_tuner import ModelTuner
from .pipeline_gate import PipelineGate
from .profiler import OptimizationProfile, OptimizationProfiler, generate_hardware_fingerprint
from .prompt_optimizer import PromptOptimizer

__all__ = [
    "ModelTuner",
    "OptimizationProfile",
    "OptimizationProfiler",
    "PipelineGate",
    "PromptOptimizer",
    "SelfOptimizer",
    "generate_hardware_fingerprint",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_optimizer/test_coordinator.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/adaptive_learning/performance/self_optimizer/ tests/unit/self_optimizer/test_coordinator.py
git commit -m "feat(self-optimizer): add SelfOptimizer coordinator"
```

---

### Task 6: Storage Integration

**Files:**
- Modify: `src/homie_core/adaptive_learning/storage.py`
- Test: `tests/unit/self_optimizer/test_storage_integration.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/self_optimizer/test_storage_integration.py
import pytest
from homie_core.adaptive_learning.storage import LearningStorage


class TestOptimizationProfileStorage:
    def test_save_and_get_profile(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        data = {"query_type": "coding", "hardware_fingerprint": "abc", "temperature": 0.4, "max_tokens": 800}
        store.save_optimization_profile("coding", "abc", data)
        result = store.get_optimization_profile("coding", "abc")
        assert result is not None
        assert result["temperature"] == 0.4

    def test_upsert_profile(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_optimization_profile("coding", "abc", {"temperature": 0.5})
        store.save_optimization_profile("coding", "abc", {"temperature": 0.3})
        result = store.get_optimization_profile("coding", "abc")
        assert result["temperature"] == 0.3

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        assert store.get_optimization_profile("unknown", "xyz") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/self_optimizer/test_storage_integration.py -v`
Expected: FAIL — methods not found

- [ ] **Step 3: Add table and methods to storage.py**

Read `src/homie_core/adaptive_learning/storage.py`. In the `initialize()` method, add after the existing `executescript`:

```sql
CREATE TABLE IF NOT EXISTS optimization_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_type TEXT NOT NULL,
    hardware_fingerprint TEXT NOT NULL,
    profile_data TEXT NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(query_type, hardware_fingerprint)
);
```

Add methods:

```python
def save_optimization_profile(self, query_type: str, hardware_fp: str, data: dict) -> None:
    """Save or update an optimization profile."""
    if self._conn is None:
        return
    with self._lock:
        self._conn.execute(
            """INSERT INTO optimization_profiles (query_type, hardware_fingerprint, profile_data, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(query_type, hardware_fingerprint) DO UPDATE SET
               profile_data = excluded.profile_data, updated_at = excluded.updated_at""",
            (query_type, hardware_fp, json.dumps(data), time.time()),
        )
        self._conn.commit()

def get_optimization_profile(self, query_type: str, hardware_fp: str) -> Optional[dict]:
    """Get an optimization profile."""
    if self._conn is None:
        return None
    row = self._conn.execute(
        "SELECT profile_data FROM optimization_profiles WHERE query_type = ? AND hardware_fingerprint = ?",
        (query_type, hardware_fp),
    ).fetchone()
    return json.loads(row["profile_data"]) if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/self_optimizer/test_storage_integration.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/adaptive_learning/storage.py tests/unit/self_optimizer/test_storage_integration.py
git commit -m "feat(self-optimizer): add optimization_profiles table to LearningStorage"
```

---

### Task 7: Config & Integration Test

**Files:**
- Modify: `src/homie_core/config.py`
- Modify: `homie.config.yaml`
- Create: `tests/unit/self_optimizer/test_config.py`
- Create: `tests/integration/test_self_optimizer_lifecycle.py`

- [ ] **Step 1: Write failing config test**

```python
# tests/unit/self_optimizer/test_config.py
import pytest
from homie_core.config import SelfOptimizerConfig


class TestSelfOptimizerConfig:
    def test_defaults(self):
        cfg = SelfOptimizerConfig()
        assert cfg.enabled is True
        assert cfg.prompt.deduplication is True
        assert cfg.model.auto_temperature is True
        assert cfg.pipeline.gating_enabled is True
        assert cfg.pipeline.promotion_threshold == 3
```

- [ ] **Step 2: Add config classes**

Read `src/homie_core/config.py`. Add before HomieConfig:

```python
class PromptOptimizerConfig(BaseModel):
    deduplication: bool = True
    relevance_threshold: float = 0.3
    history_compression: bool = True
    max_history_turns_default: int = 10

class ModelTunerConfig(BaseModel):
    auto_temperature: bool = True
    auto_context_length: bool = True
    auto_gpu_layers: bool = True
    response_length_learning: bool = True

class PipelineGateConfig(BaseModel):
    gating_enabled: bool = True
    self_correcting: bool = True
    promotion_threshold: int = 3

class SelfOptimizerConfig(BaseModel):
    enabled: bool = True
    prompt: PromptOptimizerConfig = Field(default_factory=PromptOptimizerConfig)
    model: ModelTunerConfig = Field(default_factory=ModelTunerConfig)
    pipeline: PipelineGateConfig = Field(default_factory=PipelineGateConfig)
```

Add to `AdaptiveLearningConfig`: `self_optimizer: SelfOptimizerConfig = Field(default_factory=SelfOptimizerConfig)`

Add `self_optimizer:` section nested under `adaptive_learning:` in `homie.config.yaml`.

- [ ] **Step 3: Write integration test**

```python
# tests/integration/test_self_optimizer_lifecycle.py
"""Integration test: self-optimizer lifecycle."""
import pytest
from homie_core.adaptive_learning.performance.self_optimizer.coordinator import SelfOptimizer
from homie_core.adaptive_learning.storage import LearningStorage


class TestSelfOptimizerLifecycle:
    def test_full_optimization_cycle(self, tmp_path):
        """Full flow: optimize → execute → record → profile improves."""
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="test_hw")

        # First query — uses defaults
        params = opt.optimize_query("moderate", query_hint="coding")
        assert params["effective_complexity"] == "moderate"
        assert "temperature" in params

        # Record results over multiple queries
        for i in range(10):
            opt.record_result("coding", "moderate", temperature=0.3, max_tokens=512, response_tokens=250, latency_ms=200)

        # Profile should now have learned
        profile = opt.profiler.get_profile("coding")
        assert profile is not None
        assert profile.sample_count == 10
        assert profile.avg_response_tokens > 0

    def test_pipeline_gate_promotes_on_clarifications(self, tmp_path):
        """Gate promotes tier after repeated clarification requests."""
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="hw", promotion_threshold=2)

        # Trivial queries cause clarifications
        opt.record_clarification("trivial")
        opt.record_clarification("trivial")

        # Next trivial query should be promoted
        params = opt.optimize_query("trivial")
        assert params["effective_complexity"] == "simple"

    def test_prompt_compression(self, tmp_path):
        """Prompt optimizer compresses based on complexity."""
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="hw")

        long_prompt = "System prompt.\n" + "Fact: " * 500
        opt.prompt_optimizer.set_complexity("trivial")
        compressed = opt.prompt_optimizer.modify_prompt(long_prompt)
        assert len(compressed) < len(long_prompt)
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/unit/self_optimizer/ tests/integration/test_self_optimizer_lifecycle.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full regression**

Run: `python -m pytest tests/unit/self_healing/ tests/unit/adaptive_learning/ tests/unit/knowledge_evolution/ tests/unit/self_optimizer/ tests/integration/ -q --tb=short`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/config.py homie.config.yaml tests/unit/self_optimizer/test_config.py tests/integration/test_self_optimizer_lifecycle.py
git commit -m "feat(self-optimizer): add config integration and lifecycle tests"
```
