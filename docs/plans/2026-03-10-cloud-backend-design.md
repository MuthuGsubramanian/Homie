# Cloud API Backend Design

**Date:** 2026-03-10
**Status:** Approved

## Goal

Allow Homie to use any OpenAI-compatible cloud API (OpenAI, Groq, Together, Mistral, DeepSeek, OpenRouter, etc.) as an alternative to local GGUF inference. User selects local or cloud during initial setup, and can switch at runtime.

## Architecture

A single `CloudBackend` that speaks the OpenAI-compatible `/v1/chat/completions` protocol. The existing `ModelEngine` dispatches to it when `format="cloud"`. No new dependencies — uses `urllib` (stdlib).

## Components

### 1. CloudBackend (`src/homie_core/model/cloud_backend.py`)

Implements the backend interface: `load()`, `generate()`, `stream()`, `unload()`.

- `load()` validates connection via `GET /v1/models`
- `generate()` sends `POST /v1/chat/completions` (non-streaming)
- `stream()` sends `POST /v1/chat/completions` (streaming SSE)
- Auth via `Authorization: Bearer <api_key>` header

### 2. Config Changes (`LLMConfig`)

New fields:
- `api_key: str = ""` — stored in config YAML, overridable via `HOMIE_API_KEY` env var
- `api_base_url: str = ""` — provider endpoint (e.g., `https://api.openai.com/v1`)

New backend value: `"cloud"`. When backend is cloud, `model_path` holds the model ID (e.g., `gpt-4o`).

### 3. ModelEngine.load() Update

New `elif entry.format == "cloud":` branch that instantiates `CloudBackend`.

### 4. Setup Wizard Changes (`init.py`)

After hardware detection, insert branching step: "Local model or Cloud API?"

If cloud:
1. Pick provider (OpenAI, Groq, Together, DeepSeek, OpenRouter, Custom)
2. Enter API key
3. Auto-discover models via `GET /v1/models`; fallback to preset list
4. Pick model
5. Save config with `backend: cloud`

### 5. CLI Switch Support

Cloud models registered as `ModelEntry(format="cloud", path="<model_id>")`. Existing `homie model switch` works without changes.

## Data Flow

```
User message → BrainOrchestrator → ModelEngine → CloudBackend
    → HTTP POST /v1/chat/completions {model, messages, max_tokens, temperature}
    ← Response JSON → extract content → return to user
```

## Error Handling

- Connection failures → clear error with provider URL
- 401 → "Invalid API key"
- 429 → "Rate limited"
- 404 → "Model not available"

## Testing

- Mock HTTP server unit tests for CloudBackend (generate, stream, auth, errors)
- Config serialization tests for new fields
- Engine dispatch test for format="cloud"
- Setup wizard branching test
- Model discovery + preset fallback test

## Decisions

- **Single OpenAI-compatible protocol** covers most providers
- **API keys in config file** (plaintext) for simplicity
- **No new dependencies** — stdlib urllib only
- **Auto-discover models** from API, preset list as fallback
- **User selects local/cloud at setup**, can switch via `homie model switch`
