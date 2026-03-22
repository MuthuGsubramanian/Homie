# Performance Self-Optimizer — Design Spec

**Date:** 2026-03-22
**Status:** Approved
**Sub-project:** 4 of 5 (Self-Healing → Adaptive Learning → Knowledge Evolution → **Performance Self-Optimizer** → Model Fine-Tuning)

---

## Vision

Aggressively optimize Homie's runtime performance by actively tuning prompts, model parameters, and pipeline stages. Three optimization axes work together: PromptOptimizer compresses token waste, ModelTuner auto-adjusts inference parameters, and PipelineGate skips unnecessary cognitive stages. Optimizations persist as hardware-aware profiles that improve over time.

## Design Decisions

- **Aggressive optimization** — maximize speed, rely on self-healing rollback as safety net
- **Dynamic pipeline** — skip stages AND tune parameters based on query complexity
- **Hardware-aware profiles** — optimizations keyed by (query_type, hardware_fingerprint), adapt when hardware changes
- **Persistent learning** — profiles improve over time via EMA, same pattern as PreferenceEngine
- **Self-correcting gates** — if skipping a stage hurts quality (detected via implicit signals), auto-promote

---

## 1. Architecture Overview

The Performance Self-Optimizer sits between the observation layer (sub-projects 1-2) and the execution layer:

```
Self-Healing PerformanceAnalyzer → metrics & bottleneck observations
Adaptive Learning PerformanceOptimizer → cache hits, context relevance scores
       ↓
  Performance Self-Optimizer (NEW)
  ├── PromptOptimizer — compress prompts, reduce tokens
  ├── ModelTuner — auto-adjust temperature, context_length, gpu_layers, max_tokens
  ├── PipelineGate — skip unnecessary cognitive stages
  └── OptimizationProfiler — persist learned profiles per (query_type, hardware)
       ↓
  CognitiveArchitecture, ModelEngine, InferenceRouter
```

---

## 2. Prompt Optimizer

Targets the two most variable prompt components: retrieved context and conversation history.

### Strategies

| Strategy | What it does | Savings |
|----------|-------------|---------|
| Context deduplication | Remove semantically duplicate facts/episodes | 10-30% |
| Relevance truncation | Drop context below relevance threshold (from ContextOptimizer scores) | 20-50% |
| History compression | Summarize older turns instead of full text | 30-60% on long conversations |
| Fragment caching | Cache system prompt + preference layer as precomputed block | Saves re-tokenization |
| Adaptive window | Shrink context window for simple queries | Variable |

### Token Budget by Complexity

| Complexity | Max context tokens | Max history turns | Skip retrieval? |
|------------|-------------------|-------------------|-----------------|
| Trivial | 500 | 2 | Yes |
| Simple | 1500 | 5 | No |
| Moderate | 3000 | 10 | No |
| Complex | 6000 | 20 | No |
| Deep | 12000 | All | No |

Implemented as middleware via `modify_prompt` hook. Intercepts before inference, measures raw token count, applies strategies, logs savings.

---

## 3. Model Parameter Tuner

Auto-selects optimal parameters per inference call.

### Parameters Tuned

| Parameter | Range | Determined by | Tunability |
|-----------|-------|---------------|------------|
| `temperature` | 0.1 — 1.0 | Query type: factual=low, creative=high, code=low | **Per-call** via generate() |
| `max_tokens` | 64 — 4096 | Learned average response length × 1.5 buffer | **Per-call** via generate() |
| `context_length` | 2048 — 65536 | Complexity + available VRAM | **Load-time** — requires model reload |
| `gpu_layers` | 0 — max | Available VRAM, concurrent load, priority | **Load-time** — requires model reload |

**Note:** `temperature` and `max_tokens` are tuned per inference call. `context_length` and `gpu_layers` are load-time parameters on `ModelEngine.load()` — only changed on hardware profile transitions, not per query.

### Response Length Learning

Tracks actual response lengths per query type via EMA:
- "Code question" → avg 800 tokens → max_tokens = 1200
- "Quick chat" → avg 50 tokens → max_tokens = 128

### Hardware-Aware GPU Tuning

| VRAM | Strategy |
|------|----------|
| >12GB | Full gpu_layers (-1), large context |
| 6-12GB | Calculated gpu_layers, moderate context |
| <6GB | Minimal gpu_layers, compressed context |
| None | CPU-only, smallest context, aggressive prompt compression |

Re-evaluates when VRAM pressure detected (from self-healing metrics), workload changes, or system resources shift.

### Parameter Selection Flow

1. Query arrives → classify complexity
2. Lookup optimization profile for (query_type, hardware)
3. Profile exists → use learned parameters; no profile → use complexity defaults
4. After response → record actual tokens, latency
5. Update profile via EMA

---

## 4. Pipeline Gate

### Stage Gating by Complexity

The existing CognitiveArchitecture runs 6 stages. For simple queries, most stages are waste.

| Complexity | Stages executed | Skipped |
|------------|----------------|---------|
| Trivial | CLASSIFY → REASON | PERCEIVE, RETRIEVE, REFLECT, ADAPT |
| Simple | CLASSIFY → RETRIEVE → REASON | PERCEIVE, REFLECT, ADAPT |
| Moderate | PERCEIVE → CLASSIFY → RETRIEVE → REASON → REFLECT | ADAPT |
| Complex | All 6 | None |
| Deep | All 6 | None |

### Implementation

Hooks into `PipelineStage.CLASSIFIED` via existing HookRegistry. Overrides the complexity tier returned from the hook, which cascades through the existing token budget and retrieval gating logic in CognitiveArchitecture. There are no explicit skip flags — the tier IS the gate. A trivial tier naturally skips retrieval and reflection through the existing complexity-based branching.

### Self-Correcting Gates

Learns from implicit feedback signals (sub-project 2's ObservationStream):
- Trivial-gated queries get clarification requests → promote to simple
- Simple-gated queries get good engagement → keep
- Moderate queries never use reflection output → consider demoting

Promotion threshold: 3 clarification requests before promoting a query type's tier.

### Measuring Impact

Logs per gated execution: stages skipped, total latency, quality signal. If skipping consistently hurts quality, auto-promotes.

---

## 5. Optimization Profiler

### Profile Model

```python
@dataclass
class OptimizationProfile:
    query_type: str              # "coding", "chat", "factual", "creative"
    hardware_fingerprint: str    # hash of GPU name + VRAM + RAM
    temperature: float
    max_tokens: int
    context_budget: int
    pipeline_tier: str           # complexity tier override
    avg_response_tokens: float
    avg_latency_ms: float
    sample_count: int
```

Keyed by `(query_type, hardware_fingerprint)`. Hardware fingerprint changes → profiles persist but confidence resets, parameters re-learn.

### Persistence

SQLite table `optimization_profiles` in LearningStorage:
- query_type, hardware_fingerprint, parameters (JSON), avg_response_tokens, avg_latency_ms, sample_count, updated_at
- UNIQUE(query_type, hardware_fingerprint)

---

## 6. File Structure

```
src/homie_core/adaptive_learning/performance/
├── optimizer.py              # Existing — wire self_optimizer
├── response_cache.py         # Existing
├── context_optimizer.py      # Existing
├── resource_scheduler.py     # Existing
├── self_optimizer/
│   ├── __init__.py
│   ├── prompt_optimizer.py   # PromptOptimizer — compression, dedup, history summarization
│   ├── model_tuner.py        # ModelTuner — parameter selection per query
│   ├── pipeline_gate.py      # PipelineGate — skip unnecessary cognitive stages
│   ├── profiler.py           # OptimizationProfiler — persist and query profiles
│   └── coordinator.py        # SelfOptimizer — coordinates all + profiler
```

### Integration Points

| Existing Module | Integration |
|----------------|-------------|
| `brain/cognitive_arch.py` | PipelineGate hooks into CLASSIFIED stage |
| `middleware/base.py` | PromptOptimizer registers as middleware (modify_prompt) |
| `model/engine.py` | ModelTuner adjusts parameters before generate() |
| `adaptive_learning/performance/optimizer.py` | SelfOptimizer wired as sub-component |
| `adaptive_learning/learner.py` | Expose self-optimizer |
| `hardware/detector.py` | Hardware fingerprint for profile keying |
| `self_healing/metrics.py` | Read latency/error metrics |
| `adaptive_learning/observation/stream.py` | Read implicit signals for self-correction |
| `config.py` | Add SelfOptimizerConfig |
| `homie.config.yaml` | Add self_optimizer section nested under adaptive_learning |

### Config Addition

```yaml
self_optimizer:
  enabled: true
  prompt:
    deduplication: true
    relevance_threshold: 0.3
    history_compression: true
    max_history_turns_default: 10
  model:
    auto_temperature: true
    auto_context_length: true
    auto_gpu_layers: true
    response_length_learning: true
  pipeline:
    gating_enabled: true
    self_correcting: true
    promotion_threshold: 3
```

### New SQLite Table (in LearningStorage)

| Table | Purpose |
|-------|---------|
| `optimization_profiles` | Learned parameters per (query_type, hardware_fingerprint) |
