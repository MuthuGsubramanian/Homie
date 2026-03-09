from __future__ import annotations

GPU_PROFILES = [
    {"min_vram": 16.0, "model": "Qwen2.5-72B-Instruct", "quant": "Q4_K_M", "format": "gguf", "backend": "cuda"},
    {"min_vram": 12.0, "model": "Qwen2.5-32B-Instruct", "quant": "Q4_K_M", "format": "gguf", "backend": "cuda"},
    {"min_vram": 8.0, "model": "Llama-3.1-8B-Instruct", "quant": "Q5_K_M", "format": "gguf", "backend": "cuda"},
    {"min_vram": 4.0, "model": "Phi-3-mini-4k-instruct", "quant": "Q4_K_M", "format": "gguf", "backend": "cuda"},
    {"min_vram": 0.0, "model": "Phi-3-mini-4k-instruct", "quant": "Q4_0", "format": "gguf", "backend": "cpu"},
]


def recommend_model(gpu_vram_gb: float) -> dict:
    for profile in GPU_PROFILES:
        if gpu_vram_gb >= profile["min_vram"]:
            return {
                "model": profile["model"],
                "quant": profile["quant"],
                "format": profile["format"],
                "backend": profile["backend"],
            }
    return GPU_PROFILES[-1]
