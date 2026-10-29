"""HOMIE plugin: local plan-act-observe agent loop (Agent SDK style).

Inspired by Anthropic's Agent SDK / Cursor's agent loop pattern, this plugin
gives HOMIE a self-contained "plan, act, observe" cycle that runs entirely
against the local Ollama model already configured for HOMIE (see
``homie.llm_ollama``). On each step the LLM is shown the goal plus a
transcript of prior tool calls and their results, and must return exactly
one JSON "step": a thought, an optional tool call, and whether the goal is
complete.

Tools are intentionally limited to a sandboxed workspace directory
(``~/.homie/agent_workspace`` by default) so the loop can read/write notes
and list files without ever touching the rest of the filesystem or the
network. This mirrors HOMIE's existing safety posture (see
``homie.safety``) while giving the agent loop something concrete to act on.

Enable in ``homie.config.yaml``::

    agent_sdk:
      enabled: true
      max_steps: 6
      workspace: null   # defaults to ~/.homie/agent_workspace

Use ``activate(cfg)`` to get a ready-to-run ``AgentLoopPlugin`` and call
``plugin.run("goal text")`` to execute the loop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from homie.config import HomieConfig, cfg_get
from homie.llm_ollama import LLMError, ollama_generate

DEFAULT_MAX_STEPS = 6
DEFAULT_WORKSPACE_DIRNAME = "agent_workspace"
MAX_OBSERVATION_CHARS = 4000

SCHEMA_TEXT = """{
  "thought": "string (your brief reasoning for this step)",
  "tool": "read_file|write_note|list_workspace|null",
  "args": { "optional": "object, tool-specific" },
  "done": "true|false",
  "final_answer": "string (required when done=true, else null)"
}"""


class AgentStep(BaseModel):
    """One parsed turn of the plan-act-observe loop."""

    thought: str = ""
    tool: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)
    done: bool = False
    final_answer: Optional[str] = None

    @model_validator(mode="after")
    def _final_answer_required_when_done(self):
        if self.done and not (self.final_answer and self.final_answer.strip()):
            raise ValueError("final_answer is required when done=true")
        return self


@dataclass
class ToolSpec:
    """A single local, side-effect-bounded tool the agent loop may call."""

    name: str
    description: str
    handler: Callable[[Dict[str, Any]], str]


@dataclass
class AgentRunResult:
    """Outcome of a full ``AgentLoopPlugin.run()`` call."""

    success: bool
    final_answer: Optional[str]
    steps_used: int
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def _extract_json_block(text: str) -> str:
    """Strip code fences / prose and isolate the first JSON object substring."""
    cleaned = text.strip().strip("`").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class AgentLoopPlugin:
    """Local plan-act-observe agent loop running on HOMIE's configured LLM.

    The loop is deliberately read/write-local-only: every built-in tool is
    confined to a sandbox workspace directory so the plugin is safe to run
    without any extra permission gating.
    """

    def __init__(self, cfg: HomieConfig):
        self.cfg = cfg
        self.max_steps = int(cfg_get(cfg, "agent_sdk", "max_steps", default=DEFAULT_MAX_STEPS))
        workspace_override = cfg_get(cfg, "agent_sdk", "workspace", default=None)
        self.workspace = Path(workspace_override) if workspace_override else (
            Path.home() / ".homie" / DEFAULT_WORKSPACE_DIRNAME
        )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.tools: Dict[str, ToolSpec] = {}
        self._register_builtin_tools()

    # -- tool registry -----------------------------------------------------

    def _register_builtin_tools(self) -> None:
        self.tools["read_file"] = ToolSpec(
            name="read_file",
            description="Read a text file from the agent workspace. args: {path}",
            handler=self._tool_read_file,
        )
        self.tools["write_note"] = ToolSpec(
            name="write_note",
            description="Write/overwrite a text file in the agent workspace. args: {path, content}",
            handler=self._tool_write_note,
        )
        self.tools["list_workspace"] = ToolSpec(
            name="list_workspace",
            description="List files currently in the agent workspace. args: {}",
            handler=self._tool_list_workspace,
        )

    def _resolve_in_workspace(self, raw_path: str) -> Path:
        """Resolve a user/LLM-supplied path and refuse anything outside the sandbox."""
        candidate = (self.workspace / raw_path).resolve()
        if self.workspace.resolve() not in candidate.parents and candidate != self.workspace.resolve():
            raise ValueError(f"Path '{raw_path}' escapes the agent workspace")
        return candidate

    def _tool_read_file(self, args: Dict[str, Any]) -> str:
        path = self._resolve_in_workspace(str(args.get("path", "")))
        if not path.is_file():
            return f"error: no such file '{args.get('path')}'"
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:MAX_OBSERVATION_CHARS]

    def _tool_write_note(self, args: Dict[str, Any]) -> str:
        path = self._resolve_in_workspace(str(args.get("path", "")))
        content = str(args.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {args.get('path')}"

    def _tool_list_workspace(self, args: Dict[str, Any]) -> str:
        names = sorted(p.name for p in self.workspace.iterdir())
        return json.dumps(names)

    # -- prompt / parsing ---------------------------------------------------

    def _build_prompt(self, goal: str, transcript: List[Dict[str, Any]]) -> str:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        history = "\n".join(
            f"Step {i + 1} thought: {t['thought']}\n"
            f"  tool: {t.get('tool')} args: {t.get('args')}\n"
            f"  observation: {t.get('observation')}"
            for i, t in enumerate(transcript)
        ) or "(no steps taken yet)"

        return f"""
You are HOMIE's local agent loop. Work the goal below one step at a time.
Return ONLY valid JSON, no markdown, no prose.

Schema:
{SCHEMA_TEXT}

Available tools:
{tool_lines}

Goal: {goal}

Transcript so far:
{history}

Decide the single next step. Set done=true with final_answer once the goal
is satisfied or cannot be progressed further.
""".strip()

    def _parse_step(self, raw: str) -> AgentStep:
        blob = _extract_json_block(raw)
        data = json.loads(blob)
        return AgentStep.model_validate(data)

    # -- main loop ------------------------------------------------------------

    def run(self, goal: str) -> AgentRunResult:
        """Execute the plan-act-observe loop until done or max_steps is hit."""
        transcript: List[Dict[str, Any]] = []

        for step_num in range(1, self.max_steps + 1):
            prompt = self._build_prompt(goal, transcript)
            try:
                raw = ollama_generate(self.cfg, prompt)
            except LLMError as exc:
                return AgentRunResult(
                    success=False, final_answer=None, steps_used=step_num,
                    transcript=transcript, error=f"LLM call failed: {exc}",
                )

            try:
                step = self._parse_step(raw)
            except (json.JSONDecodeError, ValidationError) as exc:
                logging.warning("Agent loop step %d produced invalid JSON: %s", step_num, exc)
                transcript.append(
                    {"thought": "(invalid step, skipped)", "tool": None, "args": {}, "observation": str(exc)}
                )
                continue

            observation = "(no tool called)"
            if step.tool:
                spec = self.tools.get(step.tool)
                if spec is None:
                    observation = f"error: unknown tool '{step.tool}'"
                else:
                    try:
                        observation = spec.handler(step.args)
                    except Exception as exc:  # noqa: BLE001 - surface as observation, keep loop alive
                        observation = f"error: {exc}"

            transcript.append(
                {"thought": step.thought, "tool": step.tool, "args": step.args, "observation": observation}
            )

            if step.done:
                return AgentRunResult(
                    success=True, final_answer=step.final_answer, steps_used=step_num, transcript=transcript
                )

        return AgentRunResult(
            success=False, final_answer=None, steps_used=self.max_steps, transcript=transcript,
            error="max_steps reached without the goal being marked done",
        )


_active_plugin: Optional[AgentLoopPlugin] = None


def activate(cfg: HomieConfig) -> AgentLoopPlugin:
    """Create and register the agent loop plugin. Idempotent per process."""
    global _active_plugin
    if not cfg_get(cfg, "agent_sdk", "enabled", default=False):
        raise RuntimeError("agent_sdk.enabled is false in homie.config.yaml")
    _active_plugin = AgentLoopPlugin(cfg)
    logging.info(
        "agent_sdk plugin activated (workspace=%s, max_steps=%d)",
        _active_plugin.workspace,
        _active_plugin.max_steps,
    )
    return _active_plugin


def deactivate() -> None:
    """Drop the active plugin instance. Does not delete the workspace directory."""
    global _active_plugin
    _active_plugin = None
    logging.info("agent_sdk plugin deactivated")


def register() -> Dict[str, Any]:
    """Return plugin metadata for HOMIE's plugin registry, if one exists."""
    return {
        "name": "agent_sdk_loop",
        "activate": activate,
        "deactivate": deactivate,
    }


__all__ = [
    "AgentStep",
    "ToolSpec",
    "AgentRunResult",
    "AgentLoopPlugin",
    "activate",
    "deactivate",
    "register",
]
