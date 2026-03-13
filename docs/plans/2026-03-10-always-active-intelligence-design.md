# Always-Active Intelligence & Enterprise Design

**Date:** 2026-03-10
**Status:** Approved

## Goal

Transform Homie from a CLI chatbot into a true always-active personal intelligent assistant with background observation, proactive intelligence, overlay UI, voice/hotkey activation, and an enterprise layer for workforce deployment with privately hosted models.

## Architecture: Always-Active Daemon

Homie runs as a background service with three activation modes:

- **Event-driven observer** — Subscribes to OS events (window focus via `SetWinEventHook` on Windows, clipboard via `AddClipboardFormatListener`). Zero polling. CPU near zero when idle. On each event, updates working memory and runs task graph inference.

- **Trigger activation** — Alt+8 global hotkey and "hey homie" wake word both raise an overlay popup. The popup is a lightweight frameless window with text input + mic button. Response renders inline, dismisses on Escape or click-outside.

- **Scheduled intelligence** — APScheduler runs periodic jobs: morning briefing at detected wake time, end-of-day digest, memory consolidation, forgetting curve decay. Pure event-driven scheduling, no polling.

Single Python process, three threads:
1. **Main thread** — Event loop (hotkey listener + overlay UI)
2. **Observer thread** — OS event hooks, context aggregation
3. **Scheduler thread** — APScheduler background scheduler

Model is not loaded until needed. On trigger, lazy-loads (cloud: instant, local GGUF: ~10s first time, then kept warm with configurable idle timeout).

## Core Intelligence Algorithms

### 1. Task Graph Inference

Builds a DAG of user tasks from observations. Each node is a "task" inferred from clusters of activity (same project files, related browser tabs, similar app switches). Edges represent dependencies and context switches.

- Groups observations by temporal proximity + semantic similarity
- Detects task boundaries when user switches to unrelated activity for >5 min
- Tracks task state: `active`, `paused`, `stuck` (no progress >15 min with frequent switches), `completed` (user closes all related windows)
- Persists across sessions in episodic memory

### 2. Proactive Retrieval Engine

On every context change (window focus, file open), silently queries:
- Semantic memory for related facts
- Episodic memory for past sessions involving this file/project/URL
- Notes plugin for related notes

Stages results in a "ready context" buffer. If user triggers Homie, context is immediately available. If not, buffer is discarded. Cost: one vector query per context switch (~5ms).

### 3. Adaptive Interruption Model

Learns when to speak up vs stay quiet:
- Tracks response rate to suggestions by time-of-day, app, and task depth
- Logistic regression (pure numpy, no sklearn) on features: `minutes_in_current_task`, `switch_frequency_last_10min`, `time_since_last_interaction`, `suggestion_category`
- Only surfaces proactive suggestions when predicted acceptance >70%
- Retrains on each feedback signal (accept/dismiss/ignore)

### 4. Cross-Session Continuity

On startup or morning briefing:
- Loads last session's task graph
- Identifies incomplete tasks (files still modified, branches not merged, tabs bookmarked)
- Generates natural language summary: "Yesterday you were working on X. You left Y uncommitted. Want to pick up?"

### 5. Morning Briefing

Triggered at user's detected usual start time (from RoutineObserver):
- Lists open tasks from task graph with priority
- Shows apps/files from previous day's session
- Calendar integration (if available)
- Weather/news (if user opts in)
- Presented via overlay popup, dismissible

### 6. End-of-Day Digest

Triggered when user's activity drops (routine pattern detection):
- Summarizes: hours worked, projects touched, files edited, commits made
- Highlights: stuck tasks, incomplete work, patterns ("You context-switched 23 times today, up from your average of 12")
- Stores as episodic memory for future reference

## Enterprise Layer

Policy-based control via `homie.enterprise.yaml`:

```yaml
org_name: "Acme Corp"
model_policy:
  allowed_backends: ["cloud"]
  endpoint: "https://models.acme.internal/v1"
  api_key_env: "ACME_AI_KEY"
  allowed_models: ["llama-3.1-70b", "qwen3.5-35b"]
plugins:
  disabled: ["browser_plugin"]
  required: ["git_plugin"]
privacy:
  data_retention_days: 90
  disable_observers: ["browsing", "social"]
  audit_log: true
policy_url: ""
```

How it works:
- On startup, checks for `homie.enterprise.yaml` in config dir
- Enterprise policy merges over personal config (enterprise wins on conflicts)
- `audit_log: true` writes every LLM query + response to append-only JSON log
- Plugin restrictions enforced at PluginManager level
- `policy_url` allows fetching updated policy from central server
- User identity field (`org_user_id`) reserved for future multi-user analytics

Designed for future admin panel — structured audit logs, policy URL refresh, all extensible without rewriting.

## Overlay Popup UI

Lightweight frameless window triggered by Alt+8 or wake word:
- Text input bar with mic button
- Streams LLM response inline
- Dismisses on Escape or click-outside
- Pre-loaded with staged context from proactive retrieval
- Falls back to terminal output if overlay fails to render

## Data Flow

```
OS Events (window/clipboard) --> Observer Thread --> Working Memory
                                      |
                                      +--> Task Graph (update nodes/edges)
                                      +--> Proactive Retrieval (stage context)

Trigger (Alt+8 / voice / schedule) --> Main Thread --> Overlay Popup
                                             |
                                             +--> Load staged context
                                             +--> Query episodic + semantic memory
                                             +--> Build prompt with full context
                                             +--> ModelEngine.generate() or .stream()
                                             +--> Render response in overlay

Scheduler --> Morning Briefing (task graph + last session)
          --> End-of-Day Digest (summarize + store episode)
          --> Memory Consolidation (forgetting curve + fact extraction)
          --> Enterprise Policy Refresh (if policy_url set)
```

## Error Handling

- **Model not available** — Queue request, show "Connecting...", retry with backoff. If local GGUF fails, offer cloud switch.
- **Observer crash** — Catch in thread, log, restart after 30s. Never crash daemon.
- **Overlay fails** — Fall back to terminal + system notification.
- **Enterprise policy fetch fails** — Use cached policy, log warning, retry next schedule.

## Performance Guardrails

- Observer thread budget: <5% CPU averaged over 60s. If exceeded, increase event debounce.
- Working memory: 1000 entry LRU cap.
- Vector queries: 50ms timeout, skip if slow.
- Model idle timeout: Unload local model after 10min inactivity to free VRAM.

## Testing Strategy

**Algorithm tests (pure logic, no OS deps):**
- Task graph: observation sequences → correct nodes, edges, state transitions
- Interruption model: feature vectors + feedback → prediction thresholds
- Proactive retrieval: context changes → correct memory queries staged
- Morning briefing: previous task graph → correct summary
- Forgetting curve: decay rates and archival timing

**Integration tests (mocked OS layer):**
- Daemon lifecycle: start → events → trigger → shutdown
- Enterprise policy: merge, enforce, audit
- Cross-session continuity: persist → reload → resume prompt

**OS-specific tests (CI-skippable):**
- Hotkey registration, window event hooks, overlay rendering

Target: <30s for algorithm test suite.

## Decisions

- **Event-driven, not polling** — Zero CPU when idle
- **Three threads, one process** — Simple, no IPC overhead
- **Lazy model loading** — Don't waste VRAM when idle
- **Pure numpy for ML** — No sklearn dependency for interruption model
- **Enterprise via policy file** — No server needed, extensible to admin panel later
- **Overlay popup** — Most polished UX, dismissible, non-intrusive
- **Alt+8 hotkey** — User specified
- **70% confidence threshold** — For proactive suggestions, tunable
- **5-minute task boundary** — For task graph segmentation, tunable
