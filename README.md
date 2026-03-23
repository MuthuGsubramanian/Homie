<p align="center">
  <img src="website/public/favicon.svg" width="80" alt="Homie AI" />
</p>

<h1 align="center">H O M I E&nbsp;&nbsp;A I</h1>

<p align="center">
  <strong>Self-Evolving Local AI Assistant</strong><br/>
  <sub>Fully local. Privacy-first. No cloud. No tracking. Just you and your homie.</sub>
</p>

<p align="center">
  <a href="https://heyhomie.app">Website</a> ·
  <a href="https://heyhomie.app/download">Download</a> ·
  <a href="https://heyhomie.app/about">About</a> ·
  <a href="https://heyhomie.app/privacy">Privacy</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue?style=flat-square" alt="Version" />
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python" />
  <img src="https://img.shields.io/badge/license-MPL--2.0-green?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/cloud_calls-0-gold?style=flat-square" alt="Cloud Calls" />
  <img src="https://img.shields.io/badge/telemetry-none-red?style=flat-square" alt="Telemetry" />
</p>

---

## Quick Start

```bash
pip install homie-ai
homie init        # detects hardware, downloads a model
homie start       # say hey
```

No accounts. No API keys. No cloud.

---

## Core Intelligence

### Self-Evolution Sub-Systems

Homie ships with 5 autonomous sub-systems that allow it to grow and adapt without manual intervention:

| Sub-System | What It Does |
|------------|-------------|
| **Self-Healing Watchdog** | Monitors system health, detects failures, and auto-repairs configuration and runtime issues. |
| **Adaptive Learning** | Learns your preferences implicitly from interactions -- adjusts tone, verbosity, and behavior over time. |
| **Meta-Learning** | Tracks which strategies work best across tasks and optimizes its own reasoning pipeline. |
| **Memory Consolidation** | Merges working, episodic, and semantic memory layers -- forgets noise, strengthens signal. |
| **Plugin Auto-Discovery** | Detects installed tools and services, auto-registers them as callable tools for the agentic loop. |

### Neural Reasoning Engine

An agentic cognitive architecture with multi-step planning, tool use, and chain-of-thought reasoning. The brain decomposes complex requests into sub-tasks, executes tools, and synthesizes results -- running entirely on local hardware.

### Local ML Pipeline

- GGUF model inference via llama.cpp with full CUDA/Metal GPU acceleration
- Automatic GPU layer calculation based on available VRAM
- Model registry with hot-swapping between local, Hugging Face, and cloud models
- Inference router with automatic fallback (local -> cloud)

### Multimodal Intelligence

- **Voice**: Wake word detection, speech-to-text (Whisper), text-to-speech (Piper/Kokoro/MeloTTS), push-to-talk and continuous conversation modes
- **Screen Reading**: OCR-based screen understanding for context-aware assistance
- **Document Ingestion**: RAG pipeline over PDFs, DOCX, EPUB, PPTX, XLSX, and web pages via vector search (ChromaDB)

---

## Production Features

| Feature | Description |
|---------|-------------|
| **Voice Pipeline** | Full duplex voice with wake word ("hey homie"), STT, TTS, and conversation mode. |
| **Desktop Notifications** | System tray integration with toast notifications (Windows/Linux/macOS). |
| **Social & Messaging** | Telegram integration, social media plugins. |
| **Email** | Gmail read/draft via OAuth (no passwords stored). |
| **Encrypted Vault** | AES-256-GCM encrypted credential storage with OS keyring integration. |
| **Background Daemon** | Runs headless as a system service for always-on intelligence. |
| **Plugin System** | 12+ built-in plugins: git, terminal, clipboard, browser, IDE, notes, health, music, and more. |
| **Backup & Restore** | Encrypted backup/restore of all memories and configuration. |

---

## System Commands

```
homie start            Start the assistant
homie init             First-time setup wizard
homie daemon           Run as background service
homie model list       List installed models
homie model add        Register a local model file
homie model switch     Switch the active model
homie plugin list      List available plugins
homie plugin enable    Enable a plugin
homie backup --to      Create encrypted backup
homie restore --from   Restore from backup
```

---

## Installation Options

```bash
pip install homie-ai              # Core (text chat, memory, plugins)
pip install homie-ai[voice]       # + Voice pipeline (STT, TTS, wake word)
pip install homie-ai[neural]      # + ONNX neural reasoning
pip install homie-ai[docs]        # + Document ingestion (PDF, DOCX, EPUB)
pip install homie-ai[app]         # + Dashboard, system tray, scheduling
pip install homie-ai[all]         # Everything
```

Platform installers (Windows MSI, Linux DEB/RPM/AppImage, macOS DMG) are available on the [releases page](https://github.com/MSG-88/Homie/releases).

---

## Model Setup

Homie uses [llama.cpp](https://github.com/ggml-org/llama.cpp) for local inference. Download the server binary from the [releases page](https://github.com/ggml-org/llama.cpp/releases) and place it in `~/.homie/llama-server/`.

Models are auto-discovered in common locations (`~/.lmstudio/models/`, `~/.homie/models/`), or register manually:

```bash
homie model add /path/to/model.gguf --name my-model --format gguf --params 35B
homie model switch my-model
```

---

## Configuration

```yaml
# homie.config.yaml

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

## Minimum Specs

| | Requirement |
|---|---|
| **CPU** | Any modern x86/ARM |
| **RAM** | 8 GB minimum |
| **GPU** | Optional -- CUDA/Metal (+10x speed) |
| **Disk** | 2 GB + model size |
| **Python** | 3.11+ |
| **Internet** | Only for install, then fully offline |

---

## License

[Mozilla Public License 2.0](LICENSE)

---

<p align="center">
  <sub>Built by <a href="https://github.com/MSG-88">MSG</a> · <a href="https://heyhomie.app">heyhomie.app</a></sub>
</p>
