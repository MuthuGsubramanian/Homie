"""HOMIE plugin: local Agent SDK bridge.

Inspired by Cursor's Agent SDK (https://cursor.com/blog/agent-sdk), this
plugin brings a bounded plan -> act -> observe loop to HOMIE: an LLM
proposes one step at a time (a thought plus an optional tool call), a
local Python callable executes that tool, and the observation feeds back
into the next step until the model emits a final answer or the step
budget runs out.

Everything here is local-first:

- The only "model" call is `homie.llm_ollama.ollama_generate`, which talks
  to the Ollama endpoint already configured in `homie.config` (no new
  network surface is introduced by this plugin).
- Built-in tools are read-only (echo, list configured SSH targets, ask
  HOMIE's planner for a *plan only* -- it is never executed here).
- A tool that can actually change machine state (`execute_plan`, which
  wraps `Orchestrator.dispatch`) is only registered when the caller
  passes an `Orchestrator` AND sets `agent_sdk.allow_execution: true` in
  homie.config.yaml. Everything else stays dry-run by construction.

This mirrors the opportunity behind the discovery: teach the Agent SDK
"plan/tool/observe" pattern (good PyMasters course material) while giving
HOMIE's own plugin architecture a first taste of the same pattern, built
entirely on HOMIE's existing local LLM and config plumbing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from homie.ai_planner import plan as planner_plan
from homie.ai_planner import plan_to_dict
from homie.config import HomieConfig, cfg_get, list_targets
from homie.llm_ollama import LLMError, ollama_generate

if TYPE_CHECKING:
    from homie.controller.orchestrator import Orchestrator


STEP_SCHEMA_TEXT = """{
  "thought": "string, your short reasoning for this step",
  "tool": "string or null, name of the tool to call next",
  "args": { "optional": "object, arguments for the tool" },
  "final_answer": "string or null, set this instead of tool when done"
}"""

ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class Tool:
    """A single local capability the agent loop may call."""

    name: str
    description: str
    handler: ToolHandler


@dataclass
class AgentStep:
    """One iteration of the plan -> act -> observe loop."""

    index: int
    thought: str
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    observation: Optional[Any] = None
    final_answer: Optional[str] = None


class AgentSDKError(Exception):
    """Raised when the agent loop cannot make progress."""


def _extract_json_object(text: str) -> str:
    """Strip code fences and isolate the first {...} JSON object."""
    cleaned = text.strip().strip("`").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class AgentSDKPlugin:
    """Runs a bounded, tool-using reasoning loop against the local LLM.

    Tools are plain Python callables registered by name. Each loop step
    asks the model for one JSON-shaped decision (think, optionally call a
    tool, or finish), executes any requested tool locally, and feeds the
    observation back in as context for the next step.
    """

    def __init__(self, cfg: HomieConfig) -> None:
        self.cfg = cfg
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """Add or replace a tool in the local registry."""
        if tool.name in self._tools:
            logging.warning("agent_sdk_bridge: overwriting tool '%s'", tool.name)
        self._tools[tool.name] = tool

    def unregister_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    def clear_tools(self) -> None:
        self._tools.clear()

    @property
    def tool_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def _catalog_text(self) -> str:
        if not self._tools:
            return "(no tools registered)"
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def _history_text(self, history: List[AgentStep]) -> str:
        if not history:
            return "(no steps yet)"
        lines = []
        for step in history:
            lines.append(
                f"Step {step.index}: thought={step.thought!r} "
                f"tool={step.tool!r} args={step.args!r} "
                f"observation={step.observation!r}"
            )
        return "\n".join(lines)

    def _build_step_prompt(self, goal: str, history: List[AgentStep]) -> str:
        return f"""
You are HOMIE's local agent loop. Decide ONE next step toward the goal.
Return ONLY valid JSON, no markdown, no prose.

Schema:
{STEP_SCHEMA_TEXT}

Rules:
- Use only the tools listed below, by exact name.
- Set "tool" to null and fill "final_answer" once the goal is satisfied.
- Keep "thought" short (one sentence).

Available tools:
{self._catalog_text()}

Goal: {goal}

Steps so far:
{self._history_text(history)}
""".strip()

    def _decide_step(self, goal: str, history: List[AgentStep]) -> Dict[str, Any]:
        prompt = self._build_step_prompt(goal, history)
        raw = ollama_generate(self.cfg, prompt)
        if not raw or not raw.strip():
            raise AgentSDKError("LLM returned an empty response for the agent step")
        try:
            return json.loads(_extract_json_object(raw))
        except json.JSONDecodeError as exc:
            raise AgentSDKError(f"Could not parse agent step JSON: {exc}") from exc

    def run(self, goal: str, max_steps: Optional[int] = None) -> List[AgentStep]:
        """Run the plan -> act -> observe loop until done or out of steps."""
        budget = max_steps or int(cfg_get(self.cfg, "agent_sdk", "max_steps", default=5))
        history: List[AgentStep] = []

        for index in range(budget):
            try:
                decision = self._decide_step(goal, history)
            except (AgentSDKError, LLMError) as exc:
                history.append(
                    AgentStep(index=index, thought="error", observation=str(exc))
                )
                break

            thought = str(decision.get("thought", ""))
            final_answer = decision.get("final_answer")
            tool_name = decision.get("tool")
            args = decision.get("args") or {}

            if final_answer:
                history.append(
                    AgentStep(index=index, thought=thought, final_answer=str(final_answer))
                )
                break

            if not tool_name:
                history.append(AgentStep(index=index, thought=thought))
                continue

            tool = self._tools.get(tool_name)
            if tool is None:
                observation = {"error": f"unknown tool '{tool_name}'"}
            else:
                try:
                    observation = tool.handler(args)
                except Exception as exc:  # noqa: BLE001 - surface as observation, not a crash
                    observation = {"error": str(exc)}

            history.append(
                AgentStep(
                    index=index,
                    thought=thought,
                    tool=tool_name,
                    args=args,
                    observation=observation,
                )
            )

        return history


def _tool_echo() -> Tool:
    return Tool(
        name="echo",
        description="Echo back the given args. Useful for testing the loop.",
        handler=lambda args: {"echoed": args},
    )


def _tool_list_targets(cfg: HomieConfig) -> Tool:
    def handler(_args: Dict[str, Any]) -> Dict[str, Any]:
        return {"targets": list(list_targets(cfg))}

    return Tool(
        name="list_targets",
        description="List HOMIE's configured SSH targets (reads local config only).",
        handler=handler,
    )


def _tool_plan_only(cfg: HomieConfig) -> Tool:
    """Wraps HOMIE's existing planner as a read-only sub-agent call.

    Demonstrates agent-calling-agent composition without ever executing
    the resulting plan -- the orchestrator is never touched here.
    """

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(args.get("prompt", ""))
        if not prompt:
            return {"error": "'prompt' arg is required"}
        try:
            plan_model = planner_plan(cfg, prompt)
        except (LLMError, ValueError) as exc:
            return {"error": str(exc)}
        return {"plan": plan_to_dict(plan_model)}

    return Tool(
        name="plan_only",
        description=(
            "Ask HOMIE's planner to draft a plan for a sub-task (JSON in, "
            "JSON plan out). Never executes the plan."
        ),
        handler=handler,
    )


def _tool_execute_plan(cfg: HomieConfig, orchestrator: "Orchestrator") -> Tool:
    """Opt-in tool that actually dispatches a plan through the orchestrator.

    Only registered by `activate()` when the caller both supplies an
    Orchestrator and sets `agent_sdk.allow_execution: true` in config.
    """

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        plan_dict = args.get("plan")
        if not isinstance(plan_dict, dict):
            return {"error": "'plan' arg must be a plan object"}
        result = orchestrator.dispatch(plan_dict, dry_run=False)
        return {"ok": result.ok, "data": result.data, "error": result.error}

    return Tool(
        name="execute_plan",
        description="Execute a plan object via HOMIE's orchestrator (opt-in, mutates state).",
        handler=handler,
    )


def activate(
    cfg: HomieConfig, orchestrator: Optional["Orchestrator"] = None
) -> AgentSDKPlugin:
    """Build and wire up the agent loop, registering only safe tools by default."""
    plugin = AgentSDKPlugin(cfg)
    plugin.register_tool(_tool_echo())
    plugin.register_tool(_tool_list_targets(cfg))
    plugin.register_tool(_tool_plan_only(cfg))

    allow_execution = bool(cfg_get(cfg, "agent_sdk", "allow_execution", default=False))
    if allow_execution and orchestrator is not None:
        plugin.register_tool(_tool_execute_plan(cfg, orchestrator))
        logging.info("agent_sdk_bridge: execute_plan tool enabled (opt-in)")
    else:
        logging.info("agent_sdk_bridge: running in read-only mode (no execute_plan tool)")

    logging.info("agent_sdk_bridge: activated with tools %s", plugin.tool_names)
    return plugin


def deactivate(plugin: AgentSDKPlugin) -> None:
    """Tear down the plugin, clearing its tool registry."""
    plugin.clear_tools()
    logging.info("agent_sdk_bridge: deactivated")


def register() -> Dict[str, Any]:
    """Plugin manifest for HOMIE's plugin loader."""
    return {
        "name": "agent_sdk_bridge",
        "version": "0.1.0",
        "description": "Local plan/act/observe agent loop modeled on Cursor's Agent SDK.",
        "activate": activate,
        "deactivate": deactivate,
    }


__all__ = [
    "Tool",
    "AgentStep",
    "AgentSDKError",
    "AgentSDKPlugin",
    "activate",
    "deactivate",
    "register",
]
