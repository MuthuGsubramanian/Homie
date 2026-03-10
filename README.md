# Homie AI

Fully local, privacy-first personal AI assistant. Runs entirely on your machine with no cloud dependencies.

## Features

- **Local inference** — loads GGUF models directly via [llama.cpp](https://github.com/ggml-org/llama.cpp) with GPU acceleration (CUDA)
- **Privacy first** — all data stays on your machine, no telemetry, no external API calls
- **Voice pipeline** — wake word detection, speech-to-text, text-to-speech (optional)
- **Plugin system** — 12 built-in plugins (system, clipboard, browser, IDE, git, health, music, notes, shortcuts, terminal, network, workflows)
- **Behavioral intelligence** — habit detection, routine observation, profile synthesis
- **Memory system** — working, episodic, and semantic memory with consolidation and forgetting

## Quickstart

### Install

```bash
pip install homie-ai
```

### Setup

```bash
# Interactive setup — detects hardware, finds/downloads a model
homie init

# Or start chatting directly (uses default model path)
homie chat
```

### Model Setup

Homie uses [llama.cpp](https://github.com/ggml-org/llama.cpp) server for inference. Download the pre-built binary for your platform from the [releases page](https://github.com/ggml-org/llama.cpp/releases) and place it in `~/.homie/llama-server/`.

For Windows with CUDA:
```bash
# Download and extract to ~/.homie/llama-server/
# The zip should contain llama-server.exe and required DLLs
```

Homie auto-discovers GGUF model files in common locations (`~/.lmstudio/models/`, `~/.homie/models/`). You can also register models manually:

```bash
homie model add /path/to/model.gguf --name my-model --format gguf --params 35B
homie model switch my-model
```

## Configuration

Homie uses `homie.config.yaml` in the working directory or `~/.homie/`:

```yaml
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

Environment variable overrides: `HOMIE_LLM_BACKEND`, `HOMIE_LLM_MODEL_PATH`, `HOMIE_LLM_GPU_LAYERS`, `HOMIE_VOICE_ENABLED`, `HOMIE_STORAGE_PATH`, `HOMIE_USER_NAME`.

## CLI Commands

```
homie start          Start the assistant (alias for chat)
homie chat           Interactive chat mode
homie init           First-time setup wizard
homie model list     List installed models
homie model add      Register a local model file
homie model switch   Switch the active model
homie plugin list    List available plugins
homie plugin enable  Enable a plugin
homie backup --to    Create encrypted backup
homie restore --from Restore from backup
```

## Optional Dependencies

```bash
pip install homie-ai[model]    # HuggingFace model downloading
pip install homie-ai[voice]    # Voice pipeline (STT, TTS, wake word)
pip install homie-ai[context]  # System context tracking
pip install homie-ai[storage]  # Vector DB and encrypted backups
pip install homie-ai[app]      # Dashboard, system tray, scheduling
pip install homie-ai[all]      # Everything
```

## Requirements

- Python 3.11+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) server binary (b8149+ for Qwen3.5 support)
- A GGUF model file (auto-detected or manually configured)
- NVIDIA GPU recommended (CUDA); CPU-only mode is supported

## License

[Mozilla Public License 2.0](LICENSE)
