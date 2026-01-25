import requests
import json
from config import HomieConfig, cfg_get

def ollama_generate(cfg: HomieConfig, prompt: str) -> str:
    base_url = cfg_get(cfg, "llm", "base_url", default="http://127.0.0.1:11434")
    model = cfg_get(cfg, "llm", "model", default="glm-4.7-flash")
    temperature = cfg_get(cfg, "llm", "temperature", default=0.2)
    max_tokens = cfg_get(cfg, "llm", "max_tokens", default=800)
    timeout = cfg_get(cfg, "llm", "timeout_sec", default=60)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            # Ollama supports num_predict (max tokens to generate)
            "num_predict": max_tokens,
        },
    }

    r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")
