from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from homie.config import HomieConfig, cfg_get
from homie.llm_ollama import LLMError, ollama_generate
from homie.utils import pretty_json


SCHEMA_TEXT = """{
  "target": "msi|lenovo|all",
  "action": "run_command|check_status|copy_file",
  "command": "string (required for run_command)",
  "reason": "string (required)",
  "args": { "optional": "object" }
}"""


class PlanModel(BaseModel):
    target: str
    action: str
    command: Optional[str] = None
    reason: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _command_required(self):
        if self.action == "run_command" and (not self.command or not self.command.strip()):
            raise ValueError("command is required for run_command")
        return self


def _extract_json_block(text: str) -> str:
    """Strip code fences and isolate the first JSON object substring."""
    cleaned = text.strip().strip("`").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _try_parse_plan(raw: str) -> PlanModel:
    blob = _extract_json_block(raw)
    data = json.loads(blob)
    return PlanModel.model_validate(data)


def _build_prompt(cfg: HomieConfig, user_prompt: str) -> str:
    allowed_actions = cfg_get(
        cfg,
        "orchestrator",
        "allowed_actions",
        default=["run_command", "check_status", "copy_file"],
    )
    targets = list(cfg_get(cfg, "ssh", "targets", default={}).keys())
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    prompt = f"""
You are HOMIE, a local machine orchestrator. Today is {now}.
Return ONLY valid JSON, no markdown, no prose.

Schema:
{SCHEMA_TEXT}

Rules:
- Use only the allowed actions and targets below.
- Keep commands minimal, safe, and concise.
- command is required for run_command.
- reason is required.

Allowed actions: {allowed_actions}
Available targets: {targets}

User request: {user_prompt}
"""
    return prompt.strip()


def plan(cfg: HomieConfig, user_prompt: str) -> PlanModel:
    prompt = _build_prompt(cfg, user_prompt)
    raw = ollama_generate(cfg, prompt)
    logging.debug("LLM primary output: %s", raw)

    if not raw or not raw.strip():
        raise ValueError(
            "LLM returned an empty response. Ensure the local Ollama server is running and the model "
            "is loaded (default glm-4.7-flash)."
        )

    try:
        return _try_parse_plan(raw)
    except (json.JSONDecodeError, ValidationError) as primary_err:
        logging.warning("Primary LLM output invalid JSON, attempting repair: %s", primary_err)

    # Repair attempt
    repair_prompt = (
        "Return ONLY valid JSON. Fix this to valid JSON with the schema: "
        f"{SCHEMA_TEXT}\nHere is the invalid output:\n{raw}"
    )
    repaired = ollama_generate(cfg, repair_prompt)
    logging.debug("LLM repair output: %s", repaired)

    if not repaired or not repaired.strip():
        raise ValueError(
            "LLM returned an empty response on repair attempt. Check Ollama availability and model health."
        )

    try:
        return _try_parse_plan(repaired)
    except (json.JSONDecodeError, ValidationError) as err:
        raise ValueError(f"Unable to parse LLM plan after repair: {err}") from err


def plan_to_dict(plan_model: PlanModel) -> Dict[str, Any]:
    return plan_model.model_dump()


__all__ = ["plan", "plan_to_dict", "PlanModel"]
