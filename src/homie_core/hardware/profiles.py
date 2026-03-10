from __future__ import annotations

from pathlib import Path

GPU_PROFILES = [
    {"min_vram": 15.0, "model": "Qwen3.5-35B-A3B", "quant": "Q4_K_M", "format": "gguf", "backend": "cuda",
     "repo_id": "lmstudio-community/Qwen3.5-35B-A3B-GGUF", "context_length": 65536},
    {"min_vram": 12.0, "model": "Qwen2.5-32B-Instruct", "quant": "Q4_K_M", "format": "gguf", "backend": "cuda",
     "repo_id": "Qwen/Qwen2.5-32B-Instruct-GGUF", "context_length": 32768},
    {"min_vram": 8.0, "model": "Llama-3.1-8B-Instruct", "quant": "Q5_K_M", "format": "gguf", "backend": "cuda",
     "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "context_length": 8192},
    {"min_vram": 4.0, "model": "Phi-3-mini-4k-instruct", "quant": "Q4_K_M", "format": "gguf", "backend": "cuda",
     "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf", "context_length": 4096},
    {"min_vram": 0.0, "model": "Phi-3-mini-4k-instruct", "quant": "Q4_0", "format": "gguf", "backend": "cpu",
     "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf", "context_length": 4096},
]

# Known local directories where GGUF models may already be downloaded
LOCAL_MODELS_DIRS = [
    Path.home() / ".lmstudio" / "models",
    Path.home() / ".homie" / "models",
]


def discover_local_model(model_name: str) -> str | None:
    """Search known local directories for an existing GGUF model file."""
    for base_dir in LOCAL_MODELS_DIRS:
        if not base_dir.exists():
            continue
        for gguf_path in base_dir.rglob("*.gguf"):
            stem = gguf_path.stem.lower()
            # Skip vision projection and embedding files
            if stem.startswith("mmproj") or "embed" in stem:
                continue
            if model_name.lower().replace("-", "").replace("_", "").replace(".", "") in stem.replace("-", "").replace("_", "").replace(".", ""):
                return str(gguf_path)
    return None


def recommend_model(gpu_vram_gb: float) -> dict:
    for profile in GPU_PROFILES:
        if gpu_vram_gb >= profile["min_vram"]:
            return {
                "model": profile["model"],
                "quant": profile["quant"],
                "format": profile["format"],
                "backend": profile["backend"],
                "repo_id": profile.get("repo_id", ""),
                "context_length": profile.get("context_length", 4096),
            }
    return {
        "model": GPU_PROFILES[-1]["model"],
        "quant": GPU_PROFILES[-1]["quant"],
        "format": GPU_PROFILES[-1]["format"],
        "backend": GPU_PROFILES[-1]["backend"],
        "repo_id": GPU_PROFILES[-1].get("repo_id", ""),
        "context_length": GPU_PROFILES[-1].get("context_length", 4096),
    }
