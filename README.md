<p align="center">
  <img src="website/public/favicon.svg" width="80" alt="Homie AI" />
</p>

<h1 align="center">H O M I E&nbsp;&nbsp;A I</h1>

<p align="center">
  <strong>Your Local AI Companion</strong><br/>
  <sub>Fully local. Privacy-first. No cloud. No tracking. Just you and your homie.</sub>
</p>

<p align="center">
  <a href="https://heyhomie.app">Website</a> ·
  <a href="https://heyhomie.app/download">Download</a> ·
  <a href="https://heyhomie.app/about">About</a> ·
  <a href="https://heyhomie.app/privacy">Privacy</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python" />
  <img src="https://img.shields.io/badge/license-MPL--2.0-green?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/cloud_calls-0-gold?style=flat-square" alt="Cloud Calls" />
  <img src="https://img.shields.io/badge/telemetry-none-red?style=flat-square" alt="Telemetry" />
</p>

---

## ▸ QUEST LOG

```bash
# 01 — Install
pip install homie-ai

# 02 — Setup (detects hardware, downloads a model)
homie init

# 03 — Say hey
homie chat
```

That's it. No accounts. No API keys. No cloud.

---

## ▸ EQUIPPED ABILITIES

| Ability | Level | Description |
|---------|-------|-------------|
| **VOICE** | `LVL MAX` | Talk to your AI. It talks back. Wake word, push-to-talk, or full conversation. |
| **MEMORY** | `LVL MAX` | Learns your habits, remembers your context. Working, episodic & semantic memory. |
| **PRIVACY** | `LVL MAX` | AES-256 vault. Zero telemetry. Your data never leaves your machine. Ever. |
| **PLUGINS** | `12 SLOTS` | Browser, clipboard, IDE, git, terminal, health, music, notes — and more. |
| **BEHAVIORAL AI** | `LVL MAX` | Habit detection, routine observation, profile synthesis. |
| **LOCAL INFERENCE** | `LVL MAX` | GGUF models via llama.cpp with full GPU acceleration (CUDA/Metal). |

---

## ▸ CHARACTER STATS

```
┌─────────────────────────────────────────────────┐
│  100% LOCAL    0 CLOUD CALLS    20+ MODULES     │
│                                                 │
│  ∞ PRIVACY     AES-256 VAULT    MPL-2.0 OPEN   │
└─────────────────────────────────────────────────┘
```

---

## ▸ SYSTEM COMMANDS

```
homie start            Start the assistant (alias for chat)
homie chat             Interactive chat mode
homie init             First-time setup wizard
homie model list       List installed models
homie model add        Register a local model file
homie model switch     Switch the active model
homie plugin list      List available plugins
homie plugin enable    Enable a plugin
homie backup --to      Create encrypted backup
homie restore --from   Restore from backup
```

---

## ▸ MODEL SETUP

Homie uses [llama.cpp](https://github.com/ggml-org/llama.cpp) for local inference. Download the server binary from the [releases page](https://github.com/ggml-org/llama.cpp/releases) and place it in `~/.homie/llama-server/`.

Models are auto-discovered in common locations (`~/.lmstudio/models/`, `~/.homie/models/`), or register manually:

```bash
homie model add /path/to/model.gguf --name my-model --format gguf --params 35B
homie model switch my-model
```

---

## ▸ CONFIGURATION

```yaml
# homie.config.yaml — in working directory or ~/.homie/

llm:
  backend: gguf
  model_path: /path/to/your/model.gguf
  context_length: 65536
  gpu_layers: -1       # -1 = offload all layers to GPU
  max_tokens: 2048
  temperature: 0.7

voice:
  enabled: false
  wake_word: "hey homie"
  mode: push_to_talk

storage:
  path: ~/.homie
```

**Environment overrides:** `HOMIE_LLM_BACKEND`, `HOMIE_LLM_MODEL_PATH`, `HOMIE_LLM_GPU_LAYERS`, `HOMIE_VOICE_ENABLED`, `HOMIE_STORAGE_PATH`, `HOMIE_USER_NAME`

---

## ▸ OPTIONAL MODULES

```bash
pip install homie-ai[model]      # HuggingFace model downloading
pip install homie-ai[voice]      # Voice pipeline (STT, TTS, wake word)    ~2 GB
pip install homie-ai[context]    # System context tracking
pip install homie-ai[storage]    # Vector DB and encrypted backups
pip install homie-ai[app]        # Dashboard, system tray, scheduling
pip install homie-ai[all]        # FULL SUIT
```

---

## ▸ MINIMUM SPECS

| | Requirement |
|---|---|
| **CPU** | Any modern x86/ARM |
| **RAM** | 8 GB minimum |
| **GPU** | Optional — CUDA/Metal (+10x speed) |
| **DISK** | 2 GB + model size |
| **PYTHON** | 3.11+ |
| **INTERNET** | Only for install, then fully offline |

---

## ▸ TECH INVENTORY

| Component | Role |
|-----------|------|
| Python | Core |
| llama.cpp | Engine |
| SQLite | Storage |
| ChromaDB | Vectors |
| Whisper | Voice |
| AES-256-GCM | Vault |

---

## ▸ LICENSE

[Mozilla Public License 2.0](LICENSE)

---

<p align="center">
  <sub>Built by <a href="https://github.com/MSG-88">MSG</a> · <a href="https://heyhomie.app">heyhomie.app</a></sub>
</p>
