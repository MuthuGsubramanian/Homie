# Model Fine-Tuning Pipeline — Design Spec

**Date:** 2026-03-23
**Status:** Approved
**Sub-project:** 5 of 5 (Self-Healing → Adaptive Learning → Knowledge Evolution → Performance Self-Optimizer → **Model Fine-Tuning Pipeline**)

---

## Vision

Homie evolves its own model over time by building and pushing custom Ollama models. The Modelfile accumulates learned preferences, knowledge, instructions, and optimal parameters from sub-projects 1-4. Base models are upgraded when newer versions become available. Training data is curated for future LoRA/adapter training.

## Design Decisions

- **Ollama-based customization** — Use Ollama's Modelfile system (FROM, SYSTEM, PARAMETER, ADAPTER) to create and push `MSG-88/Homie`
- **Layered system prompt** — Base personality + learned preferences + knowledge context + instructions + customizations, each section built from different data sources
- **Milestone-based pushes** — Push when meaningful change thresholds are crossed (new facts, preference changes, customizations)
- **Two-stage validation** — Benchmark gate (fixed test prompts) then A/B shadow testing before promoting
- **Training data collection** — SFT examples (quality-filtered) + DPO pairs (corrections) for future adapter training
- **Base model upgrades** — Periodically pull newer models, migrate customizations, validate before promoting
- **Ollama API authentication** — API key stored in Homie's encrypted vault

---

## 1. Architecture Overview

### Four Stages

**Stage 1: Data Curation** — Collects SFT examples (positive interactions) and DPO pairs (corrections) from interaction history. Quality-filtered using implicit signals from the ObservationStream.

**Stage 2: Modelfile Evolution** — Builds the Modelfile from learned state: system prompt layers, parameters from self-optimizer, base model selection. Detects when enough has changed to trigger a rebuild.

**Stage 3: Validation** — Benchmark suite (fixed test prompts, 70% min score) → A/B shadow testing (50 queries or 24h, 60% win rate required) → promotion or rejection.

**Stage 4: Model Management** — Version registry, Ollama create/push, hot swap via ModelEngine, rollback on degradation.

### Milestone Triggers

| Trigger | Threshold |
|---------|-----------|
| New facts in knowledge graph | 50+ since last push |
| Preference profile changes | 10+ updates since last push |
| New customizations | 3+ since last push |
| New base model available | Detected via `ollama pull` |
| Manual trigger | User says "update your model" |

---

## 2. Modelfile Evolution System

### Modelfile Structure

```dockerfile
FROM lfm2
SYSTEM """
[Base Personality]
You are Homie, Master's personal AI assistant. You are local, private, and evolving.

[Learned Preferences]
- Response style: concise, expert-level technical depth
- Format: prefer bullet points and code-first for coding topics
- Tone: casual during evenings, professional during work hours

[Knowledge Context]
- Master works on the Homie AI project (Python, 400+ files)
- Primary tools: PyCharm, Git, Ollama, Claude Code
- Tech stack: Python 3.11+, SQLite, ChromaDB, llama.cpp

[Instructions]
- When asked about code, show the diff first
- Morning greeting should include git activity summary
- Always respect privacy — never send data externally

[Active Customizations]
- /standup command: show git commits + calendar
- Morning briefing includes project status
"""
PARAMETER temperature 0.5
PARAMETER num_ctx 32768
PARAMETER stop "<|endoftext|>"
```

### Layered Accumulation

Each section built independently:

| Layer | Source | Updates when |
|-------|--------|-------------|
| Base Personality | Static template | Never (core identity) |
| Learned Preferences | PreferenceEngine profiles | Preference confidence crosses threshold |
| Knowledge Context | Knowledge graph top entities/relationships | Significant new knowledge |
| Instructions | Behavioral rules + user instructions | New instruction learned |
| Active Customizations | CustomizationManager | Customization added/removed |

Parameters from OptimizationProfiler: temperature, num_ctx, stop tokens.

---

## 3. Ollama Integration

### OllamaManager

Wrapper around Ollama CLI with API key authentication:

```python
class OllamaManager:
    def pull(self, model: str) -> bool
    def create(self, name: str, modelfile: Path) -> bool
    def push(self, name: str) -> bool
    def list_models(self) -> list[dict]
    def show(self, name: str) -> dict
    def remove(self, name: str) -> bool
```

All commands via `subprocess` with `@resilient` decorator. Ollama API key is read from Homie's encrypted vault and injected into the subprocess environment as `OLLAMA_API_KEY` for push operations.

### Model Lifecycle

```
1. Build Modelfile from current learned state
2. Hash comparison — has Modelfile meaningfully changed?
3. Milestone check — enough new data to justify push?
4. ollama create -f Modelfile MSG-88/Homie
5. Benchmark validation (70% min)
6. Shadow test period (50 queries or 24h)
7. Shadow passes (60%+ win rate) → ollama push MSG-88/Homie
8. Archive previous version, mark new as active
9. Failure at any point → keep current, log reason
```

### Base Model Upgrades

1. Detect newer base model available
2. `ollama pull <new_base>`
3. Rebuild Modelfile with `FROM <new_base>`, keep all learned layers
4. Full benchmark + shadow test
5. If successful → promote and push

---

## 4. Validation

### Benchmark Suite

| Category | Example | Measured |
|----------|---------|----------|
| Style adherence | "Explain Python decorators" | Matches learned verbosity/format |
| Knowledge recall | "What's my main project?" | Knows user's facts |
| Instruction following | "Use bullet points" | Follows instructions |
| Tool use format | "What time is it?" | Correct tool call syntax |
| Personality | "Good morning" | Responds as Homie |

Scored via SelfReflection confidence scorer. Pass threshold: 70%.

### A/B Shadow Testing

- New model generates responses in background (not shown to user)
- Old model serves normally
- Both scored on length, format, confidence
- Trial: 50 queries or 24 hours
- Promotion if new >= old on 60%+ of queries

---

## 5. Training Data Curation

Collects data for future LoRA adapter training:

### SFT Examples (quality-filtered)

Conversations with positive signals:
- No clarification in next turn
- Quick follow-up (engaged)
- No re-asks

Format: `{"instruction": "<system>", "input": "<user>", "output": "<response>"}`

### DPO Pairs (corrections)

User corrected Homie:
- Format: `{"prompt": "<user>", "chosen": "<good>", "rejected": "<bad>"}`

Stored in `training_data` table. Exported as JSONL when adapter training is triggered.

---

## 6. File Structure

```
src/homie_core/model_evolution/
├── __init__.py
├── ollama_manager.py          # OllamaManager — CLI wrapper
├── modelfile_builder.py       # ModelfileBuilder — assembles from learned layers
├── model_registry.py          # ModelVersion tracking
├── validator.py               # Benchmark + shadow testing
├── data_curator.py            # SFT/DPO training data collection
├── evolution_engine.py        # EvolutionEngine — coordinates pipeline
└── milestone_tracker.py       # Tracks changes, triggers milestones
```

### Integration Points

| Existing Module | Integration |
|----------------|-------------|
| `adaptive_learning/preference/engine.py` | Read profiles for preferences layer |
| `adaptive_learning/knowledge/graph/query.py` | Read entities/relationships for knowledge layer |
| `adaptive_learning/customization/manager.py` | Read customizations for instructions layer |
| `adaptive_learning/performance/self_optimizer/profiler.py` | Read parameters for PARAMETER directives |
| `adaptive_learning/observation/stream.py` | Subscribe for training data curation |
| `adaptive_learning/storage.py` | New tables: model_versions, training_data |
| `self_healing/resilience/decorator.py` | @resilient on Ollama calls |
| `brain/cognitive_arch.py` | SelfReflection for benchmark scoring |
| `model/engine.py` | Hot swap after promotion |
| `config.py` | Add ModelEvolutionConfig |
| `homie.config.yaml` | Add model_evolution section |
| `homie_app/cli.py` | Boot EvolutionEngine |

### Config

```yaml
model_evolution:
  enabled: true
  ollama:
    registry_name: "MSG-88/Homie"
    base_model: "lfm2"
  milestones:
    min_new_facts: 50
    min_preference_changes: 10
    min_new_customizations: 3
  validation:
    benchmark_min_score: 0.7
    shadow_test_queries: 50
    shadow_test_max_hours: 24
    promotion_threshold: 0.6
  data_curation:
    sft_collection: true
    dpo_collection: true
    min_sft_examples: 100
```

### New SQLite Tables

| Table | Purpose |
|-------|---------|
| `model_versions` | Version registry with status, metrics, changelog |
| `training_data` | Curated SFT/DPO examples with quality scores |
