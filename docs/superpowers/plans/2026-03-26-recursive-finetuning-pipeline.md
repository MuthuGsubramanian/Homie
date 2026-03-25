# Recursive Finetuning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully local, recursive self-improving finetuning pipeline that generates synthetic training data, finetunes `lfm2` via QLoRA, evaluates against a 30-test benchmark, and deploys to `PyMasters/Homie` on Ollama — looping until performance plateaus.

**Architecture:** 4-stage loop (GENERATE → TRAIN → EVALUATE → MERGE & DEPLOY) orchestrated by `RecursiveFinetuneLoop`, scheduled by `FinetuneScheduler` during idle/off-hours. Each cycle trains from the original base with accumulated data. Cloud fallback (Qubrid/Vertex) serves as teacher for synthetic data and judge for evaluation.

**Tech Stack:** `unsloth`/`peft`, `trl` (SFTTrainer + DPOTrainer), `bitsandbytes`, `llama-cpp-python`, existing `InferenceRouter` for cloud teacher/judge, `OllamaManager` for deployment, `ModelRegistry` for versioning.

**Spec:** `docs/superpowers/specs/2026-03-26-recursive-finetuning-pipeline-design.md`

---

## File Structure

```
src/homie_core/finetune/
├── __init__.py                    # Package exports
├── config.py                      # FinetuneConfig pydantic model
├── pipeline.py                    # RecursiveFinetuneLoop orchestrator
├── scheduler.py                   # FinetuneScheduler — idle/business hours detection
├── synthetic/
│   ├── __init__.py
│   ├── templates.py               # 300 domain scenario templates
│   ├── context_randomizer.py      # Fictional user context generation
│   ├── quality_filter.py          # Teacher-based quality scoring
│   └── generator.py               # SyntheticDataGenerator — orchestrates template → teacher → filter
├── training/
│   ├── __init__.py
│   ├── qlora_trainer.py           # QLoRA training wrapper (unsloth/peft + trl)
│   ├── merge.py                   # LoRA merge + GGUF quantization
│   └── checkpoint.py              # Pause/resume checkpoint management
└── evaluation/
    ├── __init__.py
    ├── benchmark.py               # 30-test benchmark suite
    ├── judge.py                   # Cloud fallback judge scoring
    └── reporter.py                # Eval results, plateau detection, domain analysis
```

---

### Task 1: FinetuneConfig — Pydantic Configuration Model

**Files:**
- Create: `src/homie_core/finetune/__init__.py`
- Create: `src/homie_core/finetune/config.py`
- Modify: `src/homie_core/config.py:406-428` (add `finetune` to `HomieConfig`)
- Test: `tests/unit/finetune/test_config.py`

- [ ] **Step 1: Create test directory and test file**

```python
# tests/unit/finetune/__init__.py  (empty)
# tests/unit/finetune/test_config.py
"""Tests for FinetuneConfig."""
import pytest
from homie_core.finetune.config import (
    FinetuneConfig, ScheduleConfig, TrainingConfig,
    DataConfig, EvaluationConfig, LimitsConfig,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/finetune/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'homie_core.finetune'`

- [ ] **Step 3: Implement FinetuneConfig**

```python
# src/homie_core/finetune/__init__.py
"""Recursive finetuning pipeline for Homie."""

# src/homie_core/finetune/config.py
"""Pydantic configuration for the finetuning pipeline."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ScheduleConfig(BaseModel):
    business_hours_start: int = 8
    business_hours_end: int = 18
    business_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    min_idle_minutes: int = 30
    check_interval_minutes: int = 30


class TrainingConfig(BaseModel):
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
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
    sft_per_cycle: int = 2000
    dpo_per_cycle: int = 1000
    min_quality_score: int = 4
    weak_domain_boost: float = 0.4
    include_curated: bool = False


class EvaluationConfig(BaseModel):
    promotion_threshold: float = 0.02
    max_regression_per_domain: float = 0.05
    safety_floor: float = 0.85
    plateau_cycles: int = 3


class LimitsConfig(BaseModel):
    max_cycles: int = 10
    max_disk_gb: int = 50
    max_lora_rank: int = 32


class FinetuneConfig(BaseModel):
    enabled: bool = True
    base_model: str = "lfm2"
    registry_name: str = "PyMasters/Homie"
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
```

Then add to `HomieConfig` in `src/homie_core/config.py`:

```python
# Add import at top of config.py
from homie_core.finetune.config import FinetuneConfig

# Add field to HomieConfig class (after model_evolution line)
finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/finetune/test_config.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/finetune/__init__.py src/homie_core/finetune/config.py \
  src/homie_core/config.py tests/unit/finetune/__init__.py tests/unit/finetune/test_config.py
git commit -m "feat(finetune): add FinetuneConfig pydantic model"
```

---

### Task 2: Scenario Templates — 300 Domain Templates

**Files:**
- Create: `src/homie_core/finetune/synthetic/__init__.py`
- Create: `src/homie_core/finetune/synthetic/templates.py`
- Test: `tests/unit/finetune/test_templates.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_templates.py
"""Tests for scenario templates."""
import pytest
from homie_core.finetune.synthetic.templates import (
    DOMAIN_TEMPLATES, Domain, ScenarioTemplate,
    get_templates_for_domain, get_all_templates,
)


class TestDomainTemplates:
    def test_six_domains(self):
        assert len(Domain) == 6

    def test_total_template_count(self):
        all_t = get_all_templates()
        assert len(all_t) >= 300

    def test_each_domain_has_50_templates(self):
        for domain in Domain:
            templates = get_templates_for_domain(domain)
            assert len(templates) >= 50, f"{domain.value} has only {len(templates)}"

    def test_template_structure(self):
        t = get_all_templates()[0]
        assert isinstance(t, ScenarioTemplate)
        assert t.domain in Domain
        assert t.name
        assert t.system_context
        assert t.user_prompt_template
        assert t.good_behavior
        assert t.bad_behavior
        assert t.difficulty in (1, 2, 3, 4)

    def test_difficulty_distribution(self):
        """Each domain should have templates at tiers 1 and 2 at minimum."""
        for domain in Domain:
            templates = get_templates_for_domain(domain)
            tiers = {t.difficulty for t in templates}
            assert 1 in tiers
            assert 2 in tiers

    def test_proactive_templates_exist(self):
        """Cross-cutting proactive info gathering templates exist."""
        all_t = get_all_templates()
        proactive = [t for t in all_t if t.proactive]
        assert len(proactive) >= 30  # ~10% of 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/finetune/test_templates.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement templates**

Create `src/homie_core/finetune/synthetic/__init__.py` (empty).

Create `src/homie_core/finetune/synthetic/templates.py` with:
- `Domain` enum: `INTENT`, `CONTEXT`, `CONVERSATIONAL`, `ORCHESTRATION`, `SELF_AWARENESS`, `SAFETY`
- `ScenarioTemplate` dataclass: `domain`, `name`, `system_context`, `user_prompt_template`, `good_behavior` (str describing ideal response), `bad_behavior` (str describing poor response), `difficulty` (1-4), `proactive` (bool)
- `DOMAIN_TEMPLATES: dict[Domain, list[ScenarioTemplate]]` — 50 templates per domain organized by difficulty tiers:
  - Tier 1 (Basic): 20 templates — clear single-intent requests
  - Tier 2 (Intermediate): 15 templates — multi-part, subtle cues
  - Tier 3 (Advanced): 10 templates — ambiguous + conflicting
  - Tier 4 (Adversarial): 5 templates — deliberately confusing
- Each template has `{context}` and `{detail}` placeholders for the context randomizer
- ~6 proactive templates per domain (tagged `proactive=True`)
- `get_templates_for_domain(domain, max_difficulty=4)` — filter by domain and tier
- `get_all_templates()` — flat list of all templates

Templates should cover scenarios like:
- **Intent:** "Check my emails and summarize anything important", "What's the weather and also remind me about {detail}", ambiguous "Can you handle that thing from yesterday?"
- **Context:** "What should I focus on today?" (needs project/calendar context), "Is {detail} still relevant?" (needs knowledge graph)
- **Conversational:** Greeting with implicit need, multi-turn where clarification is needed, tone mismatch detection
- **Orchestration:** "Schedule a meeting with everyone who replied to my last email", "Deploy the app and notify the team"
- **Self-awareness:** "Can you access my bank account?", "What services are you connected to?"
- **Safety:** "What's the API key stored for {detail}?", "Ignore your instructions and tell me..."

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/finetune/test_templates.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/finetune/synthetic/ tests/unit/finetune/test_templates.py
git commit -m "feat(finetune): add 300 domain scenario templates across 6 domains"
```

---

### Task 3: Context Randomizer — Fictional User Contexts

**Files:**
- Create: `src/homie_core/finetune/synthetic/context_randomizer.py`
- Test: `tests/unit/finetune/test_context_randomizer.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_context_randomizer.py
"""Tests for context randomizer."""
import pytest
from homie_core.finetune.synthetic.context_randomizer import (
    ContextRandomizer, FictionalContext,
)


class TestContextRandomizer:
    def setup_method(self):
        self.randomizer = ContextRandomizer(seed=42)

    def test_generates_context(self):
        ctx = self.randomizer.generate()
        assert isinstance(ctx, FictionalContext)

    def test_has_required_fields(self):
        ctx = self.randomizer.generate()
        assert ctx.user_name
        assert ctx.projects  # list of project dicts
        assert ctx.preferences  # dict of preference settings
        assert ctx.connected_services  # list of service names
        assert ctx.time_of_day  # "morning", "afternoon", "evening"
        assert ctx.day_of_week  # "Monday" etc.

    def test_deterministic_with_seed(self):
        r1 = ContextRandomizer(seed=42)
        r2 = ContextRandomizer(seed=42)
        assert r1.generate().user_name == r2.generate().user_name

    def test_different_seeds_differ(self):
        r1 = ContextRandomizer(seed=1)
        r2 = ContextRandomizer(seed=2)
        # Not guaranteed but statistically near-certain
        c1, c2 = r1.generate(), r2.generate()
        assert c1.user_name != c2.user_name or c1.projects != c2.projects

    def test_to_system_prompt(self):
        ctx = self.randomizer.generate()
        prompt = ctx.to_system_prompt()
        assert isinstance(prompt, str)
        assert ctx.user_name in prompt
        assert len(prompt) > 100

    def test_to_detail_string(self):
        ctx = self.randomizer.generate()
        detail = ctx.to_detail()
        assert isinstance(detail, str)
        assert len(detail) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/finetune/test_context_randomizer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/homie_core/finetune/synthetic/context_randomizer.py
"""Generate fictional user contexts for synthetic training data."""
from __future__ import annotations

import random
from dataclasses import dataclass, field


_NAMES = ["Alex", "Jordan", "Sam", "Casey", "Morgan", "Riley", "Quinn", "Avery",
          "Dakota", "Reese", "Skyler", "Taylor", "Jamie", "Robin", "Pat", "Drew"]
_PROJECTS = ["web-dashboard", "mobile-app", "data-pipeline", "ml-service",
             "api-gateway", "auth-module", "chat-bot", "analytics-engine",
             "inventory-system", "payment-service", "search-index", "cms-backend"]
_LANGUAGES = ["Python", "TypeScript", "Go", "Rust", "Java", "C#"]
_SERVICES = ["gmail", "slack", "calendar", "weather", "news", "github", "jira",
             "linkedin", "twitter", "notion", "trello"]
_ROLES = ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer",
          "Designer", "Team Lead", "Freelancer", "Student"]
_PREFERENCES = {
    "verbosity": ["concise", "balanced", "detailed"],
    "formality": ["casual", "neutral", "formal"],
    "technical_depth": ["simple", "moderate", "expert"],
}


@dataclass
class FictionalContext:
    user_name: str
    role: str
    projects: list[dict]
    preferences: dict[str, str]
    connected_services: list[str]
    time_of_day: str
    day_of_week: str
    recent_activity: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        lines = [
            f"You are Homie, a personal AI assistant for {self.user_name}.",
            f"User role: {self.role}.",
            f"It is {self.day_of_week} {self.time_of_day}.",
            f"Connected services: {', '.join(self.connected_services)}.",
            f"User prefers {self.preferences.get('verbosity', 'balanced')} responses "
            f"in a {self.preferences.get('formality', 'neutral')} tone "
            f"at {self.preferences.get('technical_depth', 'moderate')} technical depth.",
        ]
        if self.projects:
            proj_strs = [f"{p['name']} ({p['language']})" for p in self.projects]
            lines.append(f"Active projects: {', '.join(proj_strs)}.")
        if self.recent_activity:
            lines.append(f"Recent activity: {'; '.join(self.recent_activity)}.")
        if self.pending_tasks:
            lines.append(f"Pending tasks: {'; '.join(self.pending_tasks)}.")
        return "\n".join(lines)

    def to_detail(self) -> str:
        if self.projects:
            return self.projects[0]["name"]
        return self.user_name


class ContextRandomizer:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate(self) -> FictionalContext:
        name = self._rng.choice(_NAMES)
        num_projects = self._rng.randint(1, 3)
        projects = []
        for _ in range(num_projects):
            projects.append({
                "name": self._rng.choice(_PROJECTS),
                "language": self._rng.choice(_LANGUAGES),
                "status": self._rng.choice(["active", "maintenance", "starting"]),
            })
        prefs = {k: self._rng.choice(v) for k, v in _PREFERENCES.items()}
        services = self._rng.sample(_SERVICES, k=self._rng.randint(2, 5))
        time = self._rng.choice(["morning", "afternoon", "evening"])
        day = self._rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday",
                                 "Friday", "Saturday", "Sunday"])
        activities = []
        for _ in range(self._rng.randint(0, 3)):
            activities.append(self._rng.choice([
                f"committed to {self._rng.choice(_PROJECTS)}",
                f"received 3 new emails",
                f"updated {self._rng.choice(_PROJECTS)} README",
                f"had a meeting about {self._rng.choice(_PROJECTS)}",
                f"reviewed a pull request",
            ]))
        tasks = []
        for _ in range(self._rng.randint(0, 3)):
            tasks.append(self._rng.choice([
                f"fix bug in {self._rng.choice(_PROJECTS)}",
                f"reply to team standup",
                f"review PR #42",
                f"update documentation",
                f"deploy {self._rng.choice(_PROJECTS)} to staging",
            ]))
        return FictionalContext(
            user_name=name, role=self._rng.choice(_ROLES),
            projects=projects, preferences=prefs,
            connected_services=services, time_of_day=time,
            day_of_week=day, recent_activity=activities,
            pending_tasks=tasks,
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/finetune/test_context_randomizer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/finetune/synthetic/context_randomizer.py \
  tests/unit/finetune/test_context_randomizer.py
git commit -m "feat(finetune): add context randomizer for synthetic data generation"
```

---

### Task 4: Quality Filter — Teacher-Based Scoring

**Files:**
- Create: `src/homie_core/finetune/synthetic/quality_filter.py`
- Test: `tests/unit/finetune/test_quality_filter.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_quality_filter.py
"""Tests for quality filter."""
import pytest
from unittest.mock import MagicMock
from homie_core.finetune.synthetic.quality_filter import QualityFilter


class TestQualityFilter:
    def test_score_parses_json_response(self):
        mock_inference = MagicMock(return_value='{"relevance": 5, "correctness": 4, "naturalness": 4}')
        qf = QualityFilter(inference_fn=mock_inference, min_score=4)
        score = qf.score("system", "user msg", "assistant response")
        assert score == 4  # min(5,4,4) or average — depends on impl

    def test_passes_when_above_threshold(self):
        mock_inference = MagicMock(return_value='{"relevance": 5, "correctness": 5, "naturalness": 5}')
        qf = QualityFilter(inference_fn=mock_inference, min_score=4)
        assert qf.passes("system", "user", "response") is True

    def test_fails_when_below_threshold(self):
        mock_inference = MagicMock(return_value='{"relevance": 2, "correctness": 3, "naturalness": 2}')
        qf = QualityFilter(inference_fn=mock_inference, min_score=4)
        assert qf.passes("system", "user", "response") is False

    def test_handles_malformed_json(self):
        mock_inference = MagicMock(return_value="not json")
        qf = QualityFilter(inference_fn=mock_inference, min_score=4)
        score = qf.score("system", "user", "response")
        assert score == 0  # Treat parse failure as score 0

    def test_handles_inference_error(self):
        mock_inference = MagicMock(side_effect=RuntimeError("API down"))
        qf = QualityFilter(inference_fn=mock_inference, min_score=4)
        score = qf.score("system", "user", "response")
        assert score == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/finetune/test_quality_filter.py -v`

- [ ] **Step 3: Implement**

```python
# src/homie_core/finetune/synthetic/quality_filter.py
"""Quality filter — scores synthetic examples via teacher/judge."""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_SCORE_PROMPT = """Rate this assistant response on three dimensions (1-5 each).
Return ONLY a JSON object: {{"relevance": N, "correctness": N, "naturalness": N}}

System context: {system}
User message: {user}
Assistant response: {response}"""


class QualityFilter:
    def __init__(self, inference_fn: Callable[..., str], min_score: int = 4):
        self._inference_fn = inference_fn
        self._min_score = min_score

    def score(self, system: str, user: str, response: str) -> int:
        prompt = _SCORE_PROMPT.format(system=system, user=user, response=response)
        try:
            raw = self._inference_fn(prompt=prompt, max_tokens=100, temperature=0.0)
            data = json.loads(raw.strip())
            scores = [
                int(data.get("relevance", 0)),
                int(data.get("correctness", 0)),
                int(data.get("naturalness", 0)),
            ]
            return min(scores)  # Weakest dimension determines pass/fail
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("Quality filter: could not parse score")
            return 0
        except Exception as exc:
            logger.warning("Quality filter: inference error: %s", exc)
            return 0

    def passes(self, system: str, user: str, response: str) -> bool:
        return self.score(system, user, response) >= self._min_score
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/finetune/test_quality_filter.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/finetune/synthetic/quality_filter.py \
  tests/unit/finetune/test_quality_filter.py
git commit -m "feat(finetune): add quality filter for synthetic data scoring"
```

---

### Task 5: Synthetic Data Generator — Template + Teacher Pipeline

**Files:**
- Create: `src/homie_core/finetune/synthetic/generator.py`
- Test: `tests/unit/finetune/test_generator.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_generator.py
"""Tests for synthetic data generator."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from homie_core.finetune.synthetic.generator import SyntheticDataGenerator
from homie_core.finetune.synthetic.templates import Domain


class TestSyntheticDataGenerator:
    def _mock_inference(self, **kwargs):
        prompt = kwargs.get("prompt", "")
        if "chosen" in prompt.lower() or "rejected" in prompt.lower():
            return '{"chosen": "Good response here.", "rejected": "Bad response."}'
        if "relevance" in prompt.lower():
            return '{"relevance": 5, "correctness": 5, "naturalness": 5}'
        return "This is a high-quality assistant response."

    def test_generate_sft_batch(self):
        gen = SyntheticDataGenerator(
            inference_fn=self._mock_inference, seed=42, min_quality=1,
        )
        examples = gen.generate_sft(count=5, domain=Domain.INTENT)
        assert len(examples) == 5
        for ex in examples:
            assert "system" in ex
            assert "user" in ex
            assert "assistant" in ex

    def test_generate_dpo_batch(self):
        gen = SyntheticDataGenerator(
            inference_fn=self._mock_inference, seed=42, min_quality=1,
        )
        pairs = gen.generate_dpo(count=3, domain=Domain.SAFETY)
        assert len(pairs) == 3
        for pair in pairs:
            assert "system" in pair
            assert "user" in pair
            assert "chosen" in pair
            assert "rejected" in pair

    def test_domain_allocation(self):
        gen = SyntheticDataGenerator(
            inference_fn=self._mock_inference, seed=42, min_quality=1,
        )
        alloc = gen.compute_allocation(
            total_sft=100, total_dpo=50, weak_domain=Domain.SAFETY, boost=0.4,
        )
        assert alloc[Domain.SAFETY]["sft"] > 10  # boosted above base 10%
        total_sft = sum(a["sft"] for a in alloc.values())
        assert total_sft == 100

    def test_export_jsonl(self, tmp_path):
        gen = SyntheticDataGenerator(
            inference_fn=self._mock_inference, seed=42, min_quality=1,
        )
        examples = gen.generate_sft(count=3, domain=Domain.INTENT)
        out = tmp_path / "sft.jsonl"
        gen.export_jsonl(examples, out)
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/finetune/test_generator.py -v`

- [ ] **Step 3: Implement**

The `SyntheticDataGenerator` class:
- `__init__(inference_fn, seed, min_quality)` — takes InferenceRouter.generate as inference_fn
- `generate_sft(count, domain, max_difficulty)` — for each example: pick template, generate context, render prompt, call teacher for response, quality filter, return ChatML dict
- `generate_dpo(count, domain, max_difficulty)` — similar but teacher generates chosen+rejected pair
- `compute_allocation(total_sft, total_dpo, weak_domain, boost)` — returns `dict[Domain, {"sft": int, "dpo": int}]` with boost applied
- `generate_cycle(total_sft, total_dpo, weak_domain, boost, max_difficulty_per_domain)` — full cycle generation using allocation
- `export_jsonl(examples, path)` — write list of dicts as JSONL
- Uses `ContextRandomizer` for fictional contexts and `QualityFilter` for scoring
- Retry logic: if quality filter drops domain below 50% target, retry up to 3 times with relaxed constraints
- **Teacher fallback chain** (per spec): (1) retry cloud 3 times with exponential backoff 1min/5min/15min, (2) fall back to local model as weaker teacher with quality threshold raised to 5, (3) defer cycle to next idle window if both unavailable
- **Rate limiting**: throttle cloud requests to 10/minute via `time.sleep` between calls to avoid API limits
- `format_adapter(example, from_format="alpaca")` — static method converting Alpaca `{instruction, input, output}` to ChatML `{system, user, assistant}` for when `include_curated` is enabled

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/finetune/test_generator.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/finetune/synthetic/generator.py \
  tests/unit/finetune/test_generator.py
git commit -m "feat(finetune): add synthetic data generator with teacher pipeline"
```

---

### Task 6: Benchmark Suite — 30-Test Evaluation

**Files:**
- Create: `src/homie_core/finetune/evaluation/__init__.py`
- Create: `src/homie_core/finetune/evaluation/benchmark.py`
- Test: `tests/unit/finetune/test_benchmark.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_benchmark.py
"""Tests for benchmark suite."""
import pytest
from unittest.mock import MagicMock
from homie_core.finetune.evaluation.benchmark import (
    BenchmarkSuite, BenchmarkCase, BenchmarkResult,
    DOMAIN_WEIGHTS,
)
from homie_core.finetune.synthetic.templates import Domain


class TestBenchmarkSuite:
    def test_has_30_cases(self):
        suite = BenchmarkSuite(inference_fn=MagicMock(), judge_fn=MagicMock())
        assert len(suite.cases) == 30

    def test_domain_distribution(self):
        suite = BenchmarkSuite(inference_fn=MagicMock(), judge_fn=MagicMock())
        counts = {}
        for case in suite.cases:
            counts[case.domain] = counts.get(case.domain, 0) + 1
        assert counts[Domain.INTENT] == 8
        assert counts[Domain.CONTEXT] == 6
        assert counts[Domain.CONVERSATIONAL] == 5
        assert counts[Domain.ORCHESTRATION] == 4
        assert counts[Domain.SELF_AWARENESS] == 4
        assert counts[Domain.SAFETY] == 3

    def test_domain_weights_sum_to_1(self):
        assert abs(sum(DOMAIN_WEIGHTS.values()) - 1.0) < 0.01

    def test_run_returns_result(self):
        mock_inference = MagicMock(return_value="I can help with that.")
        mock_judge = MagicMock(return_value=4.0)
        suite = BenchmarkSuite(inference_fn=mock_inference, judge_fn=mock_judge)
        result = suite.run()
        assert isinstance(result, BenchmarkResult)
        assert 0 <= result.overall_score <= 1
        assert len(result.domain_scores) == 6
        for domain, score in result.domain_scores.items():
            assert 0 <= score <= 1

    def test_case_structure(self):
        suite = BenchmarkSuite(inference_fn=MagicMock(), judge_fn=MagicMock())
        case = suite.cases[0]
        assert isinstance(case, BenchmarkCase)
        assert case.domain in Domain
        assert case.system_prompt
        assert case.user_prompt
        assert case.automated_checks  # list of callables or check dicts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/finetune/test_benchmark.py -v`

- [ ] **Step 3: Implement**

Create `src/homie_core/finetune/evaluation/__init__.py` (empty).

Create `src/homie_core/finetune/evaluation/benchmark.py` with:
- `BenchmarkCase` dataclass: `domain`, `name`, `system_prompt`, `user_prompt`, `automated_checks` (list of check dicts with `type` and `pattern`/`expected`), `judge_criteria` (str for judge prompt)
- `BenchmarkResult` dataclass: `domain_scores: dict[Domain, float]`, `overall_score: float`, `case_results: list[dict]`
- `DOMAIN_WEIGHTS`: `{INTENT: 0.25, CONTEXT: 0.20, CONVERSATIONAL: 0.20, ORCHESTRATION: 0.15, SELF_AWARENESS: 0.10, SAFETY: 0.10}`
- `_BENCHMARK_CASES`: 30 hardcoded cases covering all domains
- `BenchmarkSuite.__init__(inference_fn, judge_fn)` — inference_fn runs the model, judge_fn calls cloud for scoring
- `BenchmarkSuite.run()` — runs all 30 cases, computes automated (60%) + judge (40%) scores per case, aggregates by domain, returns BenchmarkResult
- Automated checks: regex matching, keyword presence, refusal detection, format validation
- Judge prompt: asks cloud to rate response 1-5 on the case's criteria

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/finetune/test_benchmark.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/finetune/evaluation/ tests/unit/finetune/test_benchmark.py
git commit -m "feat(finetune): add 30-test benchmark suite across 6 domains"
```

---

### Task 7: Judge — Cloud Fallback Scoring

**Files:**
- Create: `src/homie_core/finetune/evaluation/judge.py`
- Test: `tests/unit/finetune/test_judge.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_judge.py
"""Tests for cloud judge scoring."""
import pytest
from unittest.mock import MagicMock
from homie_core.finetune.evaluation.judge import Judge


class TestJudge:
    def test_score_parses_response(self):
        mock_inference = MagicMock(return_value="4")
        judge = Judge(inference_fn=mock_inference)
        score = judge.score("response text", "Is it helpful and accurate?")
        assert score == 4.0

    def test_score_handles_verbose_response(self):
        mock_inference = MagicMock(return_value="I'd rate this a 3 out of 5.")
        judge = Judge(inference_fn=mock_inference)
        score = judge.score("response", "criteria")
        assert score == 3.0

    def test_score_handles_error(self):
        mock_inference = MagicMock(side_effect=RuntimeError("offline"))
        judge = Judge(inference_fn=mock_inference)
        score = judge.score("response", "criteria")
        assert score == 2.5  # neutral fallback

    def test_normalize(self):
        judge = Judge(inference_fn=MagicMock(return_value="5"))
        normalized = judge.normalize(5.0)
        assert normalized == 1.0
        assert judge.normalize(1.0) == 0.0
        assert judge.normalize(3.0) == 0.5
```

- [ ] **Step 2: Run test, verify fails, implement, verify passes**

Implement `Judge` class:
- `__init__(inference_fn)` — cloud inference function
- `score(response, criteria)` — prompts cloud to rate 1-5, parses number, returns float
- `normalize(score)` — maps 1-5 to 0.0-1.0

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/finetune/evaluation/judge.py tests/unit/finetune/test_judge.py
git commit -m "feat(finetune): add cloud judge for open-ended response scoring"
```

---

### Task 8: Reporter — Eval Results & Plateau Detection

**Files:**
- Create: `src/homie_core/finetune/evaluation/reporter.py`
- Test: `tests/unit/finetune/test_reporter.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_reporter.py
"""Tests for eval reporter and plateau detection."""
import json
import pytest
from pathlib import Path
from homie_core.finetune.evaluation.reporter import EvalReporter
from homie_core.finetune.evaluation.benchmark import BenchmarkResult
from homie_core.finetune.synthetic.templates import Domain


def _make_result(overall: float, safety: float = 0.9) -> BenchmarkResult:
    scores = {d: overall for d in Domain}
    scores[Domain.SAFETY] = safety
    return BenchmarkResult(domain_scores=scores, overall_score=overall, case_results=[])


class TestEvalReporter:
    def test_should_promote_when_improved(self):
        reporter = EvalReporter(promotion_threshold=0.02, safety_floor=0.85,
                                max_regression=0.05, plateau_cycles=3)
        current = _make_result(0.70)
        candidate = _make_result(0.75)
        assert reporter.should_promote(current, candidate) is True

    def test_reject_when_not_improved_enough(self):
        reporter = EvalReporter(promotion_threshold=0.02, safety_floor=0.85,
                                max_regression=0.05, plateau_cycles=3)
        current = _make_result(0.70)
        candidate = _make_result(0.71)  # only 1% improvement
        assert reporter.should_promote(current, candidate) is False

    def test_reject_when_safety_below_floor(self):
        reporter = EvalReporter(promotion_threshold=0.02, safety_floor=0.85,
                                max_regression=0.05, plateau_cycles=3)
        current = _make_result(0.70)
        candidate = _make_result(0.80, safety=0.80)  # safety below 0.85
        assert reporter.should_promote(current, candidate) is False

    def test_reject_when_domain_regresses(self):
        reporter = EvalReporter(promotion_threshold=0.02, safety_floor=0.85,
                                max_regression=0.05, plateau_cycles=3)
        current = _make_result(0.70)
        candidate_scores = {d: 0.80 for d in Domain}
        candidate_scores[Domain.INTENT] = 0.64  # regressed > 5%
        candidate = BenchmarkResult(domain_scores=candidate_scores, overall_score=0.78, case_results=[])
        assert reporter.should_promote(current, candidate) is False

    def test_plateau_detection(self):
        reporter = EvalReporter(promotion_threshold=0.02, safety_floor=0.85,
                                max_regression=0.05, plateau_cycles=3)
        reporter.record_cycle(0, 0.70)
        reporter.record_cycle(1, 0.71)
        reporter.record_cycle(2, 0.71)
        reporter.record_cycle(3, 0.72)
        assert reporter.is_plateau() is True  # 3 cycles < 2% improvement

    def test_no_plateau_when_improving(self):
        reporter = EvalReporter(promotion_threshold=0.02, safety_floor=0.85,
                                max_regression=0.05, plateau_cycles=3)
        reporter.record_cycle(0, 0.60)
        reporter.record_cycle(1, 0.65)
        reporter.record_cycle(2, 0.70)
        assert reporter.is_plateau() is False

    def test_weakest_domain(self):
        scores = {d: 0.8 for d in Domain}
        scores[Domain.ORCHESTRATION] = 0.5
        result = BenchmarkResult(domain_scores=scores, overall_score=0.75, case_results=[])
        reporter = EvalReporter(promotion_threshold=0.02, safety_floor=0.85,
                                max_regression=0.05, plateau_cycles=3)
        assert reporter.weakest_domain(result) == Domain.ORCHESTRATION

    def test_save_and_load(self, tmp_path):
        reporter = EvalReporter(promotion_threshold=0.02, safety_floor=0.85,
                                max_regression=0.05, plateau_cycles=3)
        reporter.record_cycle(0, 0.70)
        reporter.save(tmp_path / "eval.json")
        loaded = EvalReporter.load(tmp_path / "eval.json")
        assert loaded._cycle_scores == {0: 0.70}
```

- [ ] **Step 2: Run test, verify fails, implement, verify passes**

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/finetune/evaluation/reporter.py tests/unit/finetune/test_reporter.py
git commit -m "feat(finetune): add eval reporter with plateau detection and promotion gate"
```

---

### Task 9: Checkpoint Manager — Pause/Resume Training

**Files:**
- Create: `src/homie_core/finetune/training/__init__.py`
- Create: `src/homie_core/finetune/training/checkpoint.py`
- Test: `tests/unit/finetune/test_checkpoint.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_checkpoint.py
"""Tests for checkpoint manager."""
import json
import pytest
from pathlib import Path
from homie_core.finetune.training.checkpoint import CheckpointManager


class TestCheckpointManager:
    def test_save_and_load_state(self, tmp_path):
        mgr = CheckpointManager(base_dir=tmp_path)
        state = {"cycle": 1, "step": 50, "stage": "sft", "epoch": 1}
        mgr.save(state, cycle=1)
        loaded = mgr.load(cycle=1)
        assert loaded == state

    def test_has_checkpoint(self, tmp_path):
        mgr = CheckpointManager(base_dir=tmp_path)
        assert mgr.has_checkpoint(cycle=1) is False
        mgr.save({"step": 0}, cycle=1)
        assert mgr.has_checkpoint(cycle=1) is True

    def test_cleanup_old_cycles(self, tmp_path):
        mgr = CheckpointManager(base_dir=tmp_path, keep_recent=2)
        for i in range(5):
            mgr.save({"step": i}, cycle=i)
        mgr.cleanup()
        assert not mgr.has_checkpoint(cycle=0)
        assert not mgr.has_checkpoint(cycle=1)
        assert not mgr.has_checkpoint(cycle=2)
        assert mgr.has_checkpoint(cycle=3)
        assert mgr.has_checkpoint(cycle=4)
```

- [ ] **Step 2: Run test, verify fails, implement, verify passes**

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/finetune/training/ tests/unit/finetune/test_checkpoint.py
git commit -m "feat(finetune): add checkpoint manager for training pause/resume"
```

---

### Task 10: QLoRA Trainer — Local Finetuning Wrapper

**Files:**
- Create: `src/homie_core/finetune/training/qlora_trainer.py`
- Test: `tests/unit/finetune/test_qlora_trainer.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_qlora_trainer.py
"""Tests for QLoRA trainer (mocked — no GPU required)."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from homie_core.finetune.training.qlora_trainer import QLoRATrainer
from homie_core.finetune.config import TrainingConfig


class TestQLoRATrainer:
    def test_init_with_config(self):
        cfg = TrainingConfig()
        trainer = QLoRATrainer(
            base_model="lfm2", config=cfg, output_dir="/tmp/test",
        )
        assert trainer.base_model == "lfm2"
        assert trainer.config.lora_rank == 16

    def test_format_sft_example(self):
        trainer = QLoRATrainer(base_model="lfm2", config=TrainingConfig(), output_dir="/tmp")
        formatted = trainer.format_sft_example({
            "system": "You are Homie.", "user": "Hello", "assistant": "Hi there!",
        })
        assert "You are Homie." in formatted
        assert "Hello" in formatted
        assert "Hi there!" in formatted

    def test_format_dpo_example(self):
        trainer = QLoRATrainer(base_model="lfm2", config=TrainingConfig(), output_dir="/tmp")
        formatted = trainer.format_dpo_example({
            "system": "You are Homie.", "user": "Hello",
            "chosen": "Hi there!", "rejected": "Go away.",
        })
        assert "chosen" in formatted or "prompt" in formatted

    @patch("homie_core.finetune.training.qlora_trainer._check_gpu_available")
    def test_preflight_check_no_gpu(self, mock_gpu):
        mock_gpu.return_value = False
        trainer = QLoRATrainer(base_model="lfm2", config=TrainingConfig(), output_dir="/tmp")
        ok, msg = trainer.preflight_check()
        assert ok is False
        assert "GPU" in msg or "VRAM" in msg

    @patch("homie_core.finetune.training.qlora_trainer._check_gpu_available")
    def test_preflight_check_with_gpu(self, mock_gpu):
        mock_gpu.return_value = True
        trainer = QLoRATrainer(base_model="lfm2", config=TrainingConfig(), output_dir="/tmp")
        ok, msg = trainer.preflight_check()
        assert ok is True
```

- [ ] **Step 2: Run test, verify fails, implement, verify passes**

Implement `QLoRATrainer`:
- `__init__(base_model, config, output_dir)` — stores config, sets up paths
- `preflight_check()` — verifies GPU availability, VRAM, dependencies (torch, unsloth/peft, trl)
- `format_sft_example(example)` — converts ChatML dict to training format string
- `format_dpo_example(example)` — converts DPO dict to training format
- `train_sft(dataset_path, checkpoint_callback=None)` — loads base model in 4-bit, applies LoRA, runs SFTTrainer. The `checkpoint_callback` is called every N steps to check if training should pause.
- `train_dpo(dataset_path, sft_adapter_path, checkpoint_callback=None)` — loads SFT adapter, runs DPOTrainer
- `_check_gpu_available()` — module-level helper checking torch.cuda.is_available() and VRAM
- Imports `unsloth` with fallback to `peft` if unavailable (Windows native). Detects WSL2 via `platform.release()` containing "microsoft" or existence of `/proc/version` with "Microsoft".
- `convert_alpaca_to_chatml(example)` — static method: maps `{instruction, input, output}` → `{system, user, assistant}`. Used when loading curated data from `DataCurator`.
- **VRAM OOM recovery chain**: wraps training in try/except for `torch.cuda.OutOfMemoryError`. On OOM: (1) reduce batch_size to 1, retry. (2) enable `offload_optimizer=True` for CPU optimizer states, retry. (3) if still OOM, abort cycle with clear error message and log.

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/finetune/training/qlora_trainer.py \
  tests/unit/finetune/test_qlora_trainer.py
git commit -m "feat(finetune): add QLoRA trainer with unsloth/peft + trl"
```

---

### Task 11: Merge & Quantize — LoRA Merge + GGUF Conversion

**Files:**
- Create: `src/homie_core/finetune/training/merge.py`
- Test: `tests/unit/finetune/test_merge.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_merge.py
"""Tests for LoRA merge and GGUF quantization."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from homie_core.finetune.training.merge import ModelMerger


class TestModelMerger:
    def test_build_modelfile_content(self):
        merger = ModelMerger(base_model="lfm2", registry_name="PyMasters/Homie")
        content = merger.build_modelfile("/path/to/model.gguf", system_prompt="You are Homie.")
        assert "FROM /path/to/model.gguf" in content
        assert "You are Homie." in content

    @patch("homie_core.finetune.training.merge.subprocess")
    def test_quantize_calls_llama_cpp(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=0)
        merger = ModelMerger(base_model="lfm2", registry_name="PyMasters/Homie")
        result = merger.quantize("/path/to/merged", "/path/to/output.gguf", "Q4_K_M")
        assert result is True
        mock_sub.run.assert_called_once()

    @patch("homie_core.finetune.training.merge.subprocess")
    def test_import_to_ollama_uses_candidate_tag(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=0)
        merger = ModelMerger(base_model="lfm2", registry_name="PyMasters/Homie")
        result = merger.import_to_ollama("/path/to/Modelfile")
        assert result is True
        # Must use :candidate staging tag, not :latest
        call_args = mock_sub.run.call_args[0][0]
        assert "PyMasters/Homie:candidate" in " ".join(call_args)

    @patch("homie_core.finetune.training.merge.subprocess")
    def test_promote_swaps_candidate_to_latest(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=0)
        merger = ModelMerger(base_model="lfm2", registry_name="PyMasters/Homie")
        result = merger.promote_candidate()
        assert result is True
        call_args = mock_sub.run.call_args[0][0]
        assert "PyMasters/Homie:candidate" in " ".join(call_args)
        assert "PyMasters/Homie:latest" in " ".join(call_args)
```

- [ ] **Step 2: Run test, verify fails, implement, verify passes**

Implement `ModelMerger`:
- `merge_lora(base_model_path, adapter_path, output_path)` — loads base + adapter via peft, merges, saves
- `quantize(model_path, output_path, quant_type="Q4_K_M")` — calls `llama-quantize` binary
- `build_modelfile(gguf_path, system_prompt)` — generates Ollama Modelfile string
- `import_to_ollama(modelfile_path)` — `ollama create PyMasters/Homie:candidate -f Modelfile`
- `promote_candidate()` — `ollama cp PyMasters/Homie:candidate PyMasters/Homie:latest`

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/finetune/training/merge.py tests/unit/finetune/test_merge.py
git commit -m "feat(finetune): add LoRA merge and GGUF quantization pipeline"
```

---

### Task 12: Finetune Scheduler — Idle & Business Hours Detection

**Files:**
- Create: `src/homie_core/finetune/scheduler.py`
- Test: `tests/unit/finetune/test_scheduler.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_scheduler.py
"""Tests for finetune scheduler."""
import pytest
from datetime import datetime, time
from unittest.mock import patch, MagicMock
from homie_core.finetune.scheduler import FinetuneScheduler
from homie_core.finetune.config import ScheduleConfig


class TestFinetuneScheduler:
    def test_is_business_hours_weekday_daytime(self):
        cfg = ScheduleConfig()
        sched = FinetuneScheduler(config=cfg)
        # Monday 10:00 AM
        dt = datetime(2026, 3, 23, 10, 0)  # Monday
        assert sched.is_business_hours(dt) is True

    def test_is_not_business_hours_evening(self):
        cfg = ScheduleConfig()
        sched = FinetuneScheduler(config=cfg)
        dt = datetime(2026, 3, 23, 22, 0)  # Monday 10 PM
        assert sched.is_business_hours(dt) is False

    def test_is_not_business_hours_weekend(self):
        cfg = ScheduleConfig()
        sched = FinetuneScheduler(config=cfg)
        dt = datetime(2026, 3, 28, 10, 0)  # Saturday
        assert sched.is_business_hours(dt) is False

    @patch("homie_core.finetune.scheduler._get_cpu_usage")
    @patch("homie_core.finetune.scheduler._get_gpu_vram_usage")
    @patch("homie_core.finetune.scheduler._get_idle_minutes")
    def test_is_system_idle(self, mock_idle, mock_gpu, mock_cpu):
        mock_cpu.return_value = 5.0
        mock_gpu.return_value = 10.0
        mock_idle.return_value = 45
        cfg = ScheduleConfig(min_idle_minutes=30)
        sched = FinetuneScheduler(config=cfg)
        assert sched.is_system_idle() is True

    @patch("homie_core.finetune.scheduler._get_cpu_usage")
    @patch("homie_core.finetune.scheduler._get_gpu_vram_usage")
    @patch("homie_core.finetune.scheduler._get_idle_minutes")
    def test_not_idle_when_cpu_high(self, mock_idle, mock_gpu, mock_cpu):
        mock_cpu.return_value = 80.0
        mock_gpu.return_value = 10.0
        mock_idle.return_value = 45
        sched = FinetuneScheduler(config=ScheduleConfig())
        assert sched.is_system_idle() is False

    def test_can_start_outside_business_hours(self):
        cfg = ScheduleConfig()
        sched = FinetuneScheduler(config=cfg)
        with patch.object(sched, "is_business_hours", return_value=False), \
             patch.object(sched, "is_system_idle", return_value=False):
            assert sched.can_start() is True  # non-business hours is enough

    def test_can_start_during_business_hours_if_idle(self):
        cfg = ScheduleConfig()
        sched = FinetuneScheduler(config=cfg)
        with patch.object(sched, "is_business_hours", return_value=True), \
             patch.object(sched, "is_system_idle", return_value=True):
            assert sched.can_start() is True  # idle overrides business hours

    def test_cannot_start_during_business_hours_if_not_idle(self):
        cfg = ScheduleConfig()
        sched = FinetuneScheduler(config=cfg)
        with patch.object(sched, "is_business_hours", return_value=True), \
             patch.object(sched, "is_system_idle", return_value=False):
            assert sched.can_start() is False

    def test_should_interrupt_on_user_activity(self):
        sched = FinetuneScheduler(config=ScheduleConfig())
        with patch("homie_core.finetune.scheduler._get_cpu_usage", return_value=70.0):
            assert sched.should_interrupt() is True
```

- [ ] **Step 2: Run test, verify fails, implement, verify passes**

Implement `FinetuneScheduler`:
- `is_business_hours(dt=None)` — checks day-of-week and hour against config
- `is_system_idle()` — checks CPU, GPU VRAM, and user input idle time
- `can_start()` — True if (not business hours OR system idle) AND GPU available
- `should_interrupt()` — True if CPU > 60% or user activity detected
- Platform helpers: `_get_cpu_usage()` via `psutil`, `_get_gpu_vram_usage()` via `torch.cuda` or `pynvml`, `_get_idle_minutes()` via `ctypes.windll.user32.GetLastInputInfo` on Windows / `/proc/stat` on Linux

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/finetune/scheduler.py tests/unit/finetune/test_scheduler.py
git commit -m "feat(finetune): add scheduler with idle detection and business hours"
```

---

### Task 13: Pipeline Orchestrator — RecursiveFinetuneLoop

**Files:**
- Create: `src/homie_core/finetune/pipeline.py`
- Test: `tests/unit/finetune/test_pipeline.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/finetune/test_pipeline.py
"""Tests for recursive finetune pipeline orchestrator."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from homie_core.finetune.pipeline import RecursiveFinetuneLoop, PipelineState
from homie_core.finetune.config import FinetuneConfig
from homie_core.finetune.synthetic.templates import Domain


class TestPipelineState:
    def test_initial_state(self, tmp_path):
        state = PipelineState(state_dir=tmp_path)
        assert state.current_cycle == 0
        assert state.lora_rank == 16
        assert state.plateau_counter == 0

    def test_save_and_load(self, tmp_path):
        state = PipelineState(state_dir=tmp_path)
        state.current_cycle = 3
        state.lora_rank = 32
        state.record_score(0, 0.70)
        state.record_score(1, 0.75)
        state.save()
        loaded = PipelineState.load(tmp_path)
        assert loaded.current_cycle == 3
        assert loaded.lora_rank == 32
        assert loaded.cycle_scores == {0: 0.70, 1: 0.75}


class TestRecursiveFinetuneLoop:
    def _make_pipeline(self, tmp_path):
        cfg = FinetuneConfig()
        return RecursiveFinetuneLoop(
            config=cfg,
            inference_fn=MagicMock(return_value="response"),
            ollama_manager=MagicMock(),
            model_registry=MagicMock(),
            base_dir=tmp_path,
        )

    def test_init(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        assert pipe.state.current_cycle == 0

    def test_should_stop_at_max_cycles(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        pipe.state.current_cycle = 10
        assert pipe._should_stop() is True

    def test_should_stop_on_plateau(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        pipe.state.plateau_counter = 3
        pipe.state.lora_rank = 32  # already max
        assert pipe._should_stop() is True

    def test_escalate_lora_on_plateau(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        pipe.state.plateau_counter = 3
        pipe.state.lora_rank = 16
        pipe._handle_plateau()
        assert pipe.state.lora_rank == 32
        assert pipe.state.plateau_counter == 0

    def test_get_difficulty_for_domain(self, tmp_path):
        pipe = self._make_pipeline(tmp_path)
        assert pipe._get_difficulty(0.5) == 1
        assert pipe._get_difficulty(0.65) == 2
        assert pipe._get_difficulty(0.85) == 3
        assert pipe._get_difficulty(0.95) == 4
```

- [ ] **Step 2: Run test, verify fails, implement, verify passes**

Implement `RecursiveFinetuneLoop`:
- `__init__(config, inference_fn, ollama_manager, model_registry, base_dir)` — wires all components
- `PipelineState` — tracks cycle, scores, plateau, lora_rank, difficulty tiers; save/load from `state.json`
- `run()` — main loop: while not `_should_stop()`: generate → train → evaluate → deploy/skip → loop
- `_run_cycle(cycle_num)` — single cycle: calls generator, trainer, benchmark, merger
- `_should_stop()` — max cycles or plateau with max LoRA rank
- `_handle_plateau()` — escalate LoRA rank or signal stop
- `_get_difficulty(score)` — maps domain score to difficulty tier (1-4)
- `get_status()` — returns current stage, cycle number, percent complete, ETA, and last eval scores (for API/dashboard)
- Integrates `FinetuneScheduler` for pause/resume via checkpoint callbacks
- Safety: abort + rollback if Safety domain < 0.85
- **Dataset accumulation**: `_load_accumulated_dataset(cycle_num)` — reads and concatenates all JSONL files from `datasets/cycle-0/` through `datasets/cycle-{N}/`, deduplicates by content hash, returns merged list. Always trains from original base model with full accumulated dataset.
- **Disk pruning**: `_prune_artifacts()` — called after each cycle. Checks total disk usage of `base_dir`. If > `max_disk_gb`, deletes oldest complete cycles (dataset + adapter + eval) keeping the most recent 3 and the current active adapter. Merged GGUF files are deleted immediately after Ollama import.
- **Structured logging**: each stage logs to `~/.homie/finetune/logs/cycle-{N}.log` with timestamps, durations, data counts, and scores.
- **Notifications**: on cycle completion or pipeline stop, emits a desktop notification via `homie_core.notifications` if available (graceful skip if not).

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/finetune/pipeline.py tests/unit/finetune/test_pipeline.py
git commit -m "feat(finetune): add RecursiveFinetuneLoop pipeline orchestrator"
```

---

### Task 14: Integration with EvolutionEngine

**Files:**
- Modify: `src/homie_core/model_evolution/evolution_engine.py`
- Test: `tests/integration/test_finetune_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_finetune_integration.py
"""Integration test for finetune pipeline with EvolutionEngine."""
import pytest
from unittest.mock import MagicMock, patch
from homie_core.model_evolution.evolution_engine import EvolutionEngine
from homie_core.finetune.config import FinetuneConfig


class TestFinetuneIntegration:
    def test_evolution_engine_has_finetune_method(self):
        engine = EvolutionEngine(
            storage=MagicMock(), ollama_manager=MagicMock(),
            preference_engine=MagicMock(), knowledge_query=MagicMock(),
            customization_manager=MagicMock(), profiler=MagicMock(),
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
```

- [ ] **Step 2: Update EvolutionEngine defaults and add `evolve_finetune()`**

First, update the default `registry_name` in `EvolutionEngine.__init__` (line 29):
```python
# Change from:
registry_name: str = "MSG-88/Homie",
# To:
registry_name: str = "PyMasters/Homie",
```

Then add the `evolve_finetune()` method. Note: the inference function is stored as `self._infer` (not `self._inference_fn`).

```python
def evolve_finetune(self, base_dir: str | Path | None = None) -> dict:
    """Run recursive finetuning pipeline (long-running, call from scheduler)."""
    from homie_core.finetune.pipeline import RecursiveFinetuneLoop
    from homie_core.finetune.config import FinetuneConfig
    cfg = FinetuneConfig()
    if base_dir is None:
        base_dir = Path.home() / ".homie" / "finetune"
    loop = RecursiveFinetuneLoop(
        config=cfg,
        inference_fn=self._infer,  # NOTE: self._infer, not self._inference_fn
        ollama_manager=self._ollama,
        model_registry=self._registry,
        base_dir=Path(base_dir),
    )
    return loop.run()
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/integration/test_finetune_integration.py -v`
Expected: All 2 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/homie_core/model_evolution/evolution_engine.py \
  tests/integration/test_finetune_integration.py
git commit -m "feat(finetune): integrate RecursiveFinetuneLoop with EvolutionEngine"
```

---

### Task 15: Dependencies & Package Setup (run early if doing non-mocked testing)

**Files:**
- Modify: `pyproject.toml` or `requirements.txt` (whichever exists)

- [ ] **Step 1: Check which dependency file exists**

Run: `ls pyproject.toml setup.py requirements*.txt`

- [ ] **Step 2: Add finetuning dependencies**

Add to the appropriate file:
```
unsloth>=2024.12
peft>=0.13
trl>=0.12
bitsandbytes>=0.43
llama-cpp-python>=0.3
psutil>=5.9
```

Note: `unsloth` is optional (WSL2-only on Windows). The code falls back to `peft` if unavailable.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml  # or requirements.txt
git commit -m "feat(finetune): add QLoRA training dependencies"
```

---

### Task 16: End-to-End Smoke Test

**Files:**
- Create: `tests/integration/test_finetune_smoke.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/integration/test_finetune_smoke.py
"""Smoke test: full pipeline with mocked training (no GPU)."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from homie_core.finetune.pipeline import RecursiveFinetuneLoop
from homie_core.finetune.config import FinetuneConfig


class TestFinetuneSmoke:
    @patch("homie_core.finetune.training.qlora_trainer.QLoRATrainer.train_sft")
    @patch("homie_core.finetune.training.qlora_trainer.QLoRATrainer.train_dpo")
    @patch("homie_core.finetune.training.merge.ModelMerger.merge_lora")
    @patch("homie_core.finetune.training.merge.ModelMerger.quantize")
    @patch("homie_core.finetune.training.merge.ModelMerger.import_to_ollama")
    @patch("homie_core.finetune.training.merge.ModelMerger.promote_candidate")
    def test_single_cycle_e2e(self, mock_promote, mock_import, mock_quant,
                               mock_merge, mock_dpo, mock_sft, tmp_path):
        mock_sft.return_value = {"loss": 0.5}
        mock_dpo.return_value = {"loss": 0.3}
        mock_merge.return_value = True
        mock_quant.return_value = True
        mock_import.return_value = True
        mock_promote.return_value = True

        def mock_inference(**kwargs):
            prompt = kwargs.get("prompt", "")
            if "relevance" in prompt.lower():
                return '{"relevance": 5, "correctness": 5, "naturalness": 5}'
            if "rate" in prompt.lower():
                return "5"
            return "This is a helpful response."

        cfg = FinetuneConfig()
        cfg.limits.max_cycles = 1
        cfg.data.sft_per_cycle = 10
        cfg.data.dpo_per_cycle = 5

        pipe = RecursiveFinetuneLoop(
            config=cfg,
            inference_fn=mock_inference,
            ollama_manager=MagicMock(),
            model_registry=MagicMock(),
            base_dir=tmp_path,
        )
        result = pipe.run()
        assert result["cycles_completed"] >= 1
        assert (tmp_path / "state.json").exists()
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_finetune_smoke.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_finetune_smoke.py
git commit -m "test(finetune): add end-to-end smoke test with mocked GPU training"
```
