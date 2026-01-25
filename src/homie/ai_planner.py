import json
from config import HomieConfig, cfg_get
from llm_ollama import ollama_generate

SYSTEM_CONTRACT = """
You are HOMIE, a local machine orchestrator.
Return ONLY valid JSON. No markdown. No extra text.

Schema:
{
  "target": "msi|lenovo|all",
  "action": "run_command|check_status|copy_file",
  "command": "string (required for run_command)",
  "reason": "string",
  "args": { "optional": "object" }
}

Rules:
- Use only the allowed targets and actions.
- Keep commands safe and minimal.
"""

def plan(cfg: HomieConfig, user_prompt: str) -> dict:
    allowed_actions = cfg_get(cfg, "orchestrator", "allowed_actions", default=["run_command","check_status"])
    targets = cfg_get(cfg, "ssh", "targets", default={})
    target_names = list(targets.keys())

    prompt = f"""{SYSTEM_CONTRACT}

Allowed actions: {allowed_actions}
Available targets: {target_names}

User request: {user_prompt}
"""

    raw = ollama_generate(cfg, prompt).strip()

    # harden: model might return code fences etc.
    raw = raw.strip("` \n")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: ask model to fix JSON
        fix_prompt = f"""Return ONLY valid JSON. Fix this to valid JSON with the schema above:\n{raw}"""
        raw2 = ollama_generate(cfg, fix_prompt).strip().strip("` \n")
        obj = json.loads(raw2)

    return obj
