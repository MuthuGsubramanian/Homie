# Neural Reasoning Engine — Design Spec

**Date:** 2026-03-23
**Status:** Approved
**Sub-project:** 1 of 5 AGI series (Neural Reasoning → Autonomous Agents → Local ML Pipeline → Multimodal Intelligence → Meta-Learning)

---

## Vision

Transform Homie from a reactive assistant into a proactive, goal-directed AGI with hierarchical planning, domain expertise, and autonomous execution. A MetaAgent orchestrates specialized sub-agents that reason, research, act, and validate — all coordinated through chain-of-thought planning with recursive goal decomposition.

## Design Decisions

- **Hierarchical agent architecture** — MetaAgent delegates to specialized agents (reasoning, research, action, validation). Sub-agents can spawn further sub-goals recursively.
- **Full autonomy by default** — Homie plans and executes without asking. Autonomy level configurable during init (full/supervised/assisted).
- **Pluggable domain expertise** — Accounting, finance, legal, tax modules with jurisdiction-aware rule application.
- **Proactive autonomous intelligence** — Homie generates reports, alerts, and analyses without being prompted, learning what to produce from user patterns.
- **Document intelligence** — Deep understanding of financial statements, contracts, invoices with domain-specific reasoning.

---

## 1. Architecture Overview

### Hierarchical Agent System

```
User goal or proactive trigger
       ↓
  MetaAgent (high-level planning + orchestration)
  ├── Decomposes goal into sub-goals (tree structure)
  ├── Selects planning strategy (direct/linear/parallel/hierarchical)
  ├── Delegates to specialized agents:
  │   ├── ReasoningAgent — deep analysis, chain-of-thought, domain reasoning
  │   ├── ResearchAgent — knowledge gathering (graph, docs, email, codebase)
  │   ├── ActionAgent — executes (code, files, tools, APIs, emails)
  │   └── ValidationAgent — tests, verifies, quality checks
  ├── Monitors progress, re-plans on failure
  └── Reports results or delivers proactive output
```

### Agent Types

| Agent | Role | Access |
|-------|------|--------|
| **MetaAgent** | Planning, delegation, monitoring, re-planning | Full orchestration |
| **ReasoningAgent** | Analysis, chain-of-thought, domain expertise | Read-only + LLM inference |
| **ResearchAgent** | Knowledge gathering from all sources | Read-only across all data |
| **ActionAgent** | Executes actions on the world | Write access, @resilient wrapped |
| **ValidationAgent** | Verifies results against goals | Read-only, scoring |

### Communication

Agents communicate through an **AgentBus** (extending existing EventBus):

```python
@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    message_type: str       # "goal", "result", "query", "status", "error"
    content: dict
    priority: int
    parent_goal_id: str
    timestamp: float
```

### Autonomy Configuration (user sets during init)

```yaml
autonomy:
  level: "full"              # full, supervised, assisted
  pause_on_irreversible: false
  max_concurrent_goals: 5
  max_recursion_depth: 10
  require_approval_for:
    - financial_transactions
    - external_communications
    - destructive_file_operations
```

---

## 2. Chain-of-Thought & Planning Engine

### Thought Process

```python
@dataclass
class ThoughtChain:
    goal: str
    steps: list[ThoughtStep]
    current_step: int
    status: str              # thinking, executing, complete, failed, replanning

@dataclass
class ThoughtStep:
    reasoning: str           # why this step
    action: str              # what to do
    expected_outcome: str    # what success looks like
    agent: str               # which agent handles this
    dependencies: list[str]  # step IDs that must complete first
    result: Optional[dict]
    status: str              # pending, active, complete, failed
```

### Planning Strategies

| Strategy | When used | How it works |
|----------|-----------|-------------|
| Direct execution | Trivial goals | Skip planning, execute immediately |
| Linear plan | Simple sequential goals | Steps in order |
| Parallel plan | Independent sub-goals | Multiple agents simultaneously |
| Iterative plan | Exploratory goals | Plan → execute → evaluate → re-plan loop |
| Hierarchical plan | Complex multi-phase goals | Recursive decomposition into sub-plans |

MetaAgent selects strategy based on goal complexity (uses existing query classification).

### Re-planning

When a step fails:
1. ValidationAgent detects the issue
2. MetaAgent evaluates: retry? alternative? escalate?
3. Re-plannable → generate new plan from current state
4. Not re-plannable → report to user with progress so far
5. All re-planning logged for Meta-Learning system

### Goal Memory

```python
@dataclass
class Goal:
    id: str
    description: str
    parent_id: Optional[str]
    thought_chain: ThoughtChain
    priority: int
    created_at: float
    completed_at: Optional[float]
    outcome: Optional[str]
    lessons_learned: list[str]
```

Stored in `goals` table. Completed goals feed knowledge graph with relationship triples.

---

## 3. Domain Reasoning & Document Intelligence

### Domain Expert Modules

Pluggable domain modules for the ReasoningAgent:

| Domain | Capabilities |
|--------|-------------|
| **Accounting** | Transaction categorization, reconciliation, P&L generation, balance sheets |
| **Finance** | Budget analysis, cash flow forecasting, expense trends, financial ratios |
| **Legal** | Contract review, clause extraction, compliance checking, risk flagging |
| **Tax** | Tax calculation, deduction identification, filing prep, jurisdiction rules |
| **General business** | Invoice processing, report generation, data extraction |

### Jurisdiction Engine

```python
@dataclass
class JurisdictionContext:
    country: str
    state_province: str
    tax_regime: str           # GST, IRS, HMRC
    currency: str
    fiscal_year_start: str
    legal_framework: str
```

- Auto-detected from user profile timezone/location
- User can override or add multiple jurisdictions
- Rules stored as knowledge graph triples

### Document Processing Pipeline

```
Document arrives (PDF, XLSX, image, email attachment)
  → Format detection (existing RAG format_detector)
  → Content extraction (existing RAG parsers + enhanced OCR)
  → Structure recognition (tables, forms, invoices, contracts)
  → Domain classification (financial? legal? general?)
  → Entity extraction (amounts, dates, parties, clauses, line items)
  → Domain reasoning (apply accounting/legal/tax rules)
  → Knowledge graph update
  → Action if needed (flag risk, calculate totals, suggest next steps)
```

### Analytical Capabilities

| Task | How |
|------|-----|
| Expense analysis | Read statements + invoices, categorize, generate breakdown |
| Contract review | Extract clauses, flag risky terms, check jurisdiction compliance |
| Tax liability | Aggregate income/expenses, apply jurisdiction rules, find deductions |
| Account reconciliation | Match transactions across sources, flag discrepancies |
| Cash flow forecast | Time-series prediction from historical data |

---

## 4. Proactive Autonomous Intelligence

### Proactive Triggers

Homie monitors and acts without being asked:

| Trigger | Automatic action |
|---------|-----------------|
| End of month | Generate financial summary, expense report, P&L |
| New bank statement (email) | Auto-categorize, flag anomalies, update budget |
| Contract deadline approaching | Alert, summarize obligations, suggest actions |
| Tax filing deadline | Prepare summary, list missing docs, calculate liability |
| Unusual transaction | Immediate alert, classify risk, suggest response |
| New invoice (email) | Extract items, match to budget, flag if over |
| Weekly | Work summary, project status, upcoming deadlines |
| Daily morning | Enhanced briefing: emails + calendar + tasks + financial alerts |
| New important email | Summarize, extract action items, queue response |

### Proactive Task Model

```python
@dataclass
class ProactiveTask:
    id: str
    trigger_type: str       # schedule, event, threshold, pattern
    trigger_config: dict
    action: str
    domain: str
    priority: int
    last_run: Optional[float]
    enabled: bool
```

### Auto-Generation Pipeline

```
Trigger fires
  → MetaAgent receives proactive task
  → Plans: gather data → analyze → generate → deliver
  → ReasoningAgent applies domain expertise
  → ActionAgent generates output
  → Delivers via: console, email digest, notification
  → Logs to knowledge graph
  → Learns from user reaction (read? acted? dismissed?)
```

### Self-Learned Proactive Behaviors

Homie learns what to generate by observing:
- What reports user manually requests → auto-generate
- What patterns trigger concern → monitor and alert
- What recurring questions → pre-compute answers
- What deadlines tracked → auto-remind with context

Feeds from Adaptive Learning ObservationStream.

---

## 5. File Structure

```
src/homie_core/neural/
├── __init__.py
├── meta_agent.py               # MetaAgent — top-level orchestrator
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # BaseAgent interface
│   ├── reasoning_agent.py      # Chain-of-thought, domain reasoning
│   ├── research_agent.py       # Knowledge gathering
│   ├── action_agent.py         # Execute actions
│   └── validation_agent.py     # Verify results
├── planning/
│   ├── __init__.py
│   ├── goal.py                 # Goal, ThoughtChain, ThoughtStep
│   ├── planner.py              # Planning strategies
│   ├── replanner.py            # Re-planning on failure
│   └── goal_memory.py          # Goal persistence
├── reasoning/
│   ├── __init__.py
│   ├── chain_of_thought.py     # Structured thinking pipeline
│   ├── domain_expert.py        # Pluggable domain modules
│   └── jurisdiction.py         # Jurisdiction-aware rules
├── proactive/
│   ├── __init__.py
│   ├── trigger_engine.py       # Proactive task triggers
│   ├── auto_generator.py       # Autonomous report/alert generation
│   └── learned_triggers.py     # Self-learned behaviors
├── communication/
│   ├── __init__.py
│   ├── agent_bus.py            # Inter-agent message bus
│   └── task_queue.py           # Priority work distribution
└── config.py                   # NeuralConfig with autonomy settings
```

### Integration Points

| Existing Module | Integration |
|----------------|-------------|
| `brain/cognitive_arch.py` | MetaAgent wraps for LLM reasoning |
| `brain/orchestrator.py` | ActionAgent uses for inference |
| `brain/tool_registry.py` | ActionAgent accesses all tools |
| `intelligence/proactive_engine.py` | Extended with trigger_engine |
| `intelligence/planner.py` | Superseded by `neural/planning/planner.py` — existing planner is deprecated |
| `intelligence/morning_briefing.py` | Enhanced with financial/legal/email |
| `adaptive_learning/knowledge/graph/` | ReasoningAgent queries + writes |
| `adaptive_learning/observation/stream.py` | Proactive triggers subscribe |
| `adaptive_learning/performance/self_optimizer/` | Agents use optimized inference |
| `self_healing/watchdog.py` | Register neural health probe |
| `self_healing/resilience/decorator.py` | @resilient on all agent actions |
| `email/` | ResearchAgent reads, ActionAgent drafts |
| `rag/pipeline.py` | Document processing for domain reasoning |
| `context/aggregator.py` | Real-time context feeds MetaAgent |
| `model/engine.py` | All agents share inference |
| `homie_app/cli.py` | Boot neural system |

### Config

```yaml
neural:
  enabled: true
  autonomy:
    level: "full"
    pause_on_irreversible: false
    max_concurrent_goals: 5
    max_recursion_depth: 10
    require_approval_for:
      - financial_transactions
      - external_communications
      - destructive_file_operations
  planning:
    default_strategy: "hierarchical"
    replan_on_failure: true
    max_replans: 3
  domains:
    accounting: true
    finance: true
    legal: true
    tax: true
    jurisdiction: "auto"
  proactive:
    enabled: true
    financial_reports: true
    email_intelligence: true
    deadline_tracking: true
    anomaly_alerts: true
    learn_from_user: true
```

### New SQLite Tables

| Table | Purpose |
|-------|---------|
| `goals` | Goal lifecycle — id, description, parent_id, status, thought_chain, outcome, lessons |
| `agent_messages` | Inter-agent communication log |
| `proactive_tasks` | Registered proactive triggers and schedules |
| `domain_rules` | Jurisdiction-specific rules for finance/legal/tax |
