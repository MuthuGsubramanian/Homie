# Homie AI — Local Personal AI Assistant Design

## Overview

Homie AI is a fully local, privacy-first personal AI assistant that runs on consumer GPUs (RTX 4080 / 16GB VRAM). It deeply understands its user — professionally, personally, and behaviorally — through passive observation, active feedback, and living memory. It listens via wake word, acts as a proactive co-pilot, and learns when to help and when to stay quiet.

**Replaces** the existing Homie project (Home Orchestrated Machine Intelligence Engine).

**Architecture:** Framework (`homie-core`) + Application (`homie-app`), distributable to any user on any hardware.

---

## 1. System Architecture

```
homie-core (framework/library)
├── Model Engine (GGUF/SafeTensors inference)
├── Brain (orchestrator, planner, suggestion engine)
├── Memory (working, episodic, semantic + consolidation)
├── Feedback (5 channels + intelligent loop)
├── Voice Pipeline (wake word, STT, TTS — optional)
├── Context Engine (screen, apps, files, clipboard)
├── Behavioral Intelligence (6 passive observers)
├── Plugin System (built-in + MCP host + custom API)
├── Storage (SQLite + ChromaDB + encrypted backup)
└── Hardware Auto-Configurator

homie-app (the application)
├── System Tray UI + Hotkeys
├── Setup Wizard (GUI + CLI)
├── 25+ Default Plugins
├── Default Personality & Prompts
├── Installer (Windows .exe)
└── CLI (homie command)
```

---

## 2. Integrated Model Engine

No external dependencies (no Ollama). Models loaded directly.

### Dual Backend

| Format | Backend | Use Case |
|--------|---------|----------|
| GGUF (Q4/Q5/Q8) | `llama-cpp-python` (CUDA) | Default — best perf on consumer GPU |
| SafeTensors (FP16/AWQ/GPTQ) | `transformers` + `accelerate` | Full precision or alt quantization |

### Hardware Auto-Select

| VRAM | Model | Quant | Format |
|------|-------|-------|--------|
| >=16GB | Qwen2.5-72B | Q4_K_M | GGUF |
| >=12GB | Qwen2.5-32B | Q4_K_M | GGUF |
| >=8GB | Llama-3.1-8B | Q5_K_M | GGUF |
| >=4GB | Phi-3-mini-4k | Q4_K_M | GGUF |
| CPU only | Phi-3-mini-4k | Q4_0 | GGUF |

### Model Management

```bash
homie model list
homie model download <repo_id>
homie model add /path/to/model.gguf
homie model remove <name>
homie model switch <name>         # hot-swap
homie model benchmark             # test speed
```

### VRAM Budget (RTX 4080)

```
LLM (70B Q4)         ~10-12 GB
faster-whisper         ~1.5 GB
ChromaDB embeddings    ~0.5 GB
Headroom               ~2-3 GB
Total                  ~14-16 GB
```

---

## 3. Living Memory System

Three layers inspired by human cognition.

### Working Memory (seconds)
- Current conversation, screen state, active app, recent clipboard
- In-RAM only, refreshes every few seconds, holds ~5 min context

### Episodic Memory (events)
- "At 3pm user had a frustrating debugging session on auth module"
- Each episode: `what happened`, `mood`, `outcome`, `context_tags`, `timestamp`
- Stored in ChromaDB as embeddings for semantic retrieval

### Semantic Memory (facts)
- Distilled from episodes and explicit teaching
- `{fact, confidence (0-1), source_count, last_confirmed, context_tags}`
- User's profile — the assistant's deep understanding

### Memory Consolidation

```
Real-time → Working Memory
              ↓ (session ends)
         Session Digest → Episodic Memory
                            ↓ (daily synthesis)
                       Patterns → Semantic Memory → Beliefs updated
```

### Memory Intelligence
- **Relevance Ranker:** hybrid retrieval (vector + recency + confidence)
- **Forgetting Curve:** memories decay if not reinforced, archived (never deleted)
- **Connection Builder:** links related memories into knowledge graphs

### Transparency
- "What do you know about me?" — shows full profile
- "Why did you suggest that?" — traces to source memories
- "Forget everything about X" — purges across all layers
- Memory dashboard in tray app

---

## 4. Intelligent Feedback Loop

### Five Input Channels

1. **Correction Tracker** — logs "no, I meant X", adjusts after 3 similar corrections
2. **Preference Learner** — implicit signals (accept/dismiss), time-of-day patterns, decays old preferences
3. **Explicit Teaching** — "remember that I'm allergic to peanuts", supports "forget this"
4. **Satisfaction Signals** — thumbs up/down, "redo differently", "don't suggest this again"
5. **Onboarding Interview** — progressive first-week guided conversation about user's life

### Five Intelligence Layers

1. **Pattern Detector** — clusters corrections, finds temporal/context correlations
2. **Belief System** — weighted beliefs with confidence scores, decay over time
3. **Contradiction Resolver** — context splits, temporal evolution, asks user when ambiguous
4. **Confidence Scorer** — pre-action scoring, low confidence = ask first, adaptive thresholds
5. **Self-Reflection Engine** — daily/weekly meta-analysis, acceptance rate trends, gap detection, surfaces insights to user

---

## 5. Behavioral Intelligence & User Profiling

Six passive observers building a holistic profile.

### Observers

| Observer | Tracks | Builds |
|----------|--------|--------|
| Media | Music played, movies, genres, skip rate | Taste profile, contextual preferences |
| Browsing | Sites visited, time spent, searches, reading depth | Interest graph, learning patterns |
| Work | Apps, files, git, terminal, coding velocity | Skill profile, productivity patterns |
| Social | Communication frequency, response time, platforms | Relationship map, style per person |
| Routine | Wake/sleep, breaks, meals, weekly cycles | Daily/weekly rhythm model |
| Emotional | Typing speed, app switching, message tone, voice tone | Mood baseline, frustration/flow detection |

### User Profile Domains

- **Identity:** name, language, locale
- **Music:** genres, artists, contextual listening, skip triggers
- **Entertainment:** movies, TV, gaming, YouTube topics
- **Work:** role, languages, frameworks, peak hours, coding style
- **Communication:** style, response speed, contacts, platforms
- **Routine:** weekday/weekend patterns, seasonal shifts
- **Health:** break preferences, fatigue signals, exercise
- **Emotional:** frustration/flow signals, mood influences
- **Interests:** topics, learning goals, reading preferences
- **Preferences:** food, weather, shopping habits
- **Repeated Actions:** daily rituals, weekly habits, trigger-response pairs

### Habit & Ritual Detection

- Action chains (A → B → C on 80% of mornings → morning ritual)
- Trigger-response pairs (event X → action Y)
- Time-locked behaviors (every Tuesday 3pm)
- Absence detection (missed routine → gentle nudge)
- Automation offers ("Want me to auto-start music when you open IDE?")

### Privacy

- Opt-in per observer
- Local-only, aggregated (not raw logs)
- Inspectable, erasable, privacy-tagged

---

## 6. Voice Pipeline (Optional Module)

### Components

| Component | Library | VRAM/CPU |
|-----------|---------|----------|
| Wake Word | OpenWakeWord ("Hey Homie") | CPU ~2% |
| STT | faster-whisper (large-v3) | ~1.5GB VRAM |
| TTS | Piper | CPU only |
| Audio I/O | PyAudio + sounddevice | Minimal |

### Voice Modes

- **Always-listening** — wake word activates STT
- **Push-to-talk** — hotkey held to speak
- **Text-only** — voice disabled
- **Quick toggle:** F9 (on/off), full mode selection in tray settings

### Voice Intelligence
- Tone detection from Whisper output → mood signal
- Voice conversations stored as episodes
- Interruption handling — stops TTS immediately when user speaks

---

## 7. Context Engine

### Components

- **Screen Monitor:** Win32 APIs, active window title/process, optional screenshot (user-consented)
- **App Tracker:** usage duration, switch frequency, deep work vs multitasking detection
- **File Indexer:** watchdog on configured dirs, incremental embedding into ChromaDB
- **Clipboard:** history tracking, smart paste suggestions
- **Aggregator:** merges all signals into unified snapshot → feeds Working Memory

All context is privacy-tagged by source and type.

---

## 8. Plugin System

### Architecture

- **Plugin Manager:** discovers, loads, hot-reloads, manages permissions, sandboxed execution
- **Plugin API:** `HomiePlugin` base class with `on_activate`, `on_context`, `on_query`, `on_action`, `on_deactivate`
- **MCP Host:** any MCP-compatible server auto-discovered as a plugin
- **Custom plugins:** community-developed, installable via CLI or tray UI

### Built-in Plugins (Phase 1 — ship with v1.0)

| Category | Plugins |
|----------|---------|
| Communication | Email, WhatsApp, Telegram, Discord, Teams |
| Productivity | Calendar, Notes, Tasks/Todo, PDF/Docs, Spreadsheets |
| Development | IDE, Git, Terminal, Docker, Databases |
| Browser | History, Open Tabs, Bookmarks, Downloads |
| System | App Tracker, Clipboard, Notifications, Power, Network, Storage |
| Media | Spotify/Music, Screenshots, Camera (presence) |
| Personal | Health, Finance, Learning, Fitness, Recipes/Meals |
| Automation | Shortcuts, Cron Jobs, Workflows |

### Plugin Management

```bash
homie plugin list
homie plugin enable <name>
homie plugin disable <name>
homie plugin mcp add <server>
```

---

## 9. Setup & Distribution

### Installation Methods

| Method | Audience |
|--------|----------|
| `.exe` installer (GUI wizard) | General users |
| `homie init` (CLI) | Power users |
| `homie init --auto` | Full auto-detection |
| `pip install homie-ai` | Developers |
| `homie-core` (lib only) | Framework/plugin developers |

### First Run Experience

1. "Hi, I'm Homie. Let me set things up."
2. Auto-detect hardware → show results
3. Download optimal model (progress bar, resume-capable)
4. Mic test (if available)
5. "What should I call you?"
6. Plugin selection (checkboxes)
7. Onboarding interview begins (progressive, skippable)
8. Tray icon appears → ready

### Backup & Restore

```bash
homie backup --to /path/to/backup    # AES-256 encrypted
homie restore --from /path/to/backup  # re-runs hardware detection
```

---

## 10. Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Inference | llama-cpp-python (CUDA), transformers + accelerate |
| Model Format | GGUF, SafeTensors |
| Model Download | huggingface_hub |
| STT | faster-whisper |
| Wake Word | OpenWakeWord |
| TTS | Piper |
| Audio I/O | PyAudio, sounddevice |
| Web/API | FastAPI |
| Vector Store | ChromaDB |
| Database | SQLite (WAL) |
| File Watching | watchdog |
| Screen/OS | pywin32, psutil |
| Encryption | cryptography (Fernet/AES-256) |
| Scheduling | APScheduler |
| MCP | mcp-python-sdk |
| Hotkeys | pynput / keyboard |
| Tray UI | pystray + Pillow |
| GUI Wizard | tkinter |
| Packaging | PyInstaller + Inno Setup |
| Testing | pytest + pytest-asyncio |

---

## 11. Project Structure

```
homie-ai/
├── pyproject.toml
├── LICENSE
├── README.md
├── docs/plans/
│
├── src/
│   ├── homie_core/
│   │   ├── brain/
│   │   │   ├── orchestrator.py
│   │   │   ├── planner.py
│   │   │   └── suggestion_engine.py
│   │   ├── model/
│   │   │   ├── engine.py
│   │   │   ├── gguf_backend.py
│   │   │   ├── transformers_backend.py
│   │   │   ├── registry.py
│   │   │   ├── downloader.py
│   │   │   ├── server.py
│   │   │   └── vram.py
│   │   ├── memory/
│   │   │   ├── working.py
│   │   │   ├── episodic.py
│   │   │   ├── semantic.py
│   │   │   ├── consolidator.py
│   │   │   ├── forgetting.py
│   │   │   └── connections.py
│   │   ├── feedback/
│   │   │   ├── collector.py
│   │   │   ├── patterns.py
│   │   │   ├── beliefs.py
│   │   │   ├── contradictions.py
│   │   │   ├── adapter.py
│   │   │   └── reflection.py
│   │   ├── behavioral/
│   │   │   ├── media_observer.py
│   │   │   ├── browsing_observer.py
│   │   │   ├── work_observer.py
│   │   │   ├── social_observer.py
│   │   │   ├── routine_observer.py
│   │   │   ├── emotional_observer.py
│   │   │   ├── habit_detector.py
│   │   │   └── profile_synthesizer.py
│   │   ├── voice/
│   │   │   ├── wakeword.py
│   │   │   ├── stt.py
│   │   │   ├── tts.py
│   │   │   ├── audio_io.py
│   │   │   └── vad.py
│   │   ├── context/
│   │   │   ├── screen_monitor.py
│   │   │   ├── app_tracker.py
│   │   │   ├── file_indexer.py
│   │   │   ├── clipboard.py
│   │   │   └── aggregator.py
│   │   ├── plugins/
│   │   │   ├── manager.py
│   │   │   ├── base.py
│   │   │   ├── permissions.py
│   │   │   ├── sandbox.py
│   │   │   └── mcp_host.py
│   │   ├── storage/
│   │   │   ├── database.py
│   │   │   ├── vectors.py
│   │   │   ├── backup.py
│   │   │   └── migrations/
│   │   ├── hardware/
│   │   │   ├── detector.py
│   │   │   ├── configurator.py
│   │   │   └── profiles.py
│   │   ├── config.py
│   │   ├── safety.py
│   │   └── utils.py
│   │
│   └── homie_app/
│       ├── cli.py
│       ├── init.py
│       ├── tray/
│       │   ├── app.py
│       │   └── dashboard.py
│       ├── wizard/
│       │   ├── gui.py
│       │   └── steps.py
│       ├── plugins/
│       │   ├── email_plugin.py
│       │   ├── calendar_plugin.py
│       │   ├── browser_plugin.py
│       │   ├── ide_plugin.py
│       │   ├── git_plugin.py
│       │   ├── terminal_plugin.py
│       │   ├── notes_plugin.py
│       │   ├── tasks_plugin.py
│       │   ├── system_plugin.py
│       │   ├── health_plugin.py
│       │   ├── music_plugin.py
│       │   ├── whatsapp_plugin.py
│       │   ├── telegram_plugin.py
│       │   ├── discord_plugin.py
│       │   ├── teams_plugin.py
│       │   ├── docs_plugin.py
│       │   ├── spreadsheets_plugin.py
│       │   ├── docker_plugin.py
│       │   ├── database_plugin.py
│       │   ├── clipboard_plugin.py
│       │   ├── notifications_plugin.py
│       │   ├── network_plugin.py
│       │   ├── finance_plugin.py
│       │   ├── learning_plugin.py
│       │   ├── shortcuts_plugin.py
│       │   └── workflows_plugin.py
│       ├── prompts/
│       │   ├── system.py
│       │   ├── onboarding.py
│       │   └── reflection.py
│       └── installer/
│           ├── build_exe.py
│           └── inno_setup.iss
│
├── plugins/
│   └── example_plugin/
│
└── tests/
    ├── unit/
    └── integration/
```

### Data Directory

```
~/.homie/
├── config.yaml
├── homie.db          # SQLite
├── chroma/           # ChromaDB
├── models/           # Downloaded GGUF/SafeTensors
├── plugins/          # Community plugins
├── workflows/
├── backups/
└── logs/
```

---

## 12. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| No Ollama | Direct GGUF/SafeTensors | Zero external deps, full control, portable |
| Monolithic service | Single process + modules | Lowest latency, simplest deploy, can split later |
| ChromaDB for episodes | Vector store | Semantic search over memories |
| SQLite for facts | Relational DB | Structured queries, beliefs, profiles |
| OpenWakeWord over Porcupine | Free, customizable | No API key, custom wake phrase |
| Piper over Coqui | Faster, more stable | Better for real-time response |
| llama-cpp-python primary | GGUF inference | Best consumer GPU performance |
| Privacy tags over access control | Flexible | Context-appropriate surfacing without rigid walls |
| Framework + App split | Reusability | Other devs can build on homie-core |
