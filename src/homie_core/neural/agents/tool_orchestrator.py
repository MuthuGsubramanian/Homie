"""ToolOrchestrator — chains multiple tools for complex multi-step tasks.

The orchestrator plans and executes sequences of tool calls, passing
results between steps and handling dependencies.  It delegates to the
ToolRegistry for individual tool execution and uses an LLM inference
function to dynamically plan tool chains from natural-language goals.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

from homie_core.brain.tool_registry import ToolCall, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Used to resolve ``$step.<index>`` references inside argument values.
_STEP_REF_PATTERN = re.compile(r"\$step\.(\d+)")


class ToolOrchestrator:
    """Chains multiple tools for complex multi-step tasks.

    Parameters
    ----------
    tool_registry:
        The :class:`ToolRegistry` that holds available tools.
    inference_fn:
        Callable with signature ``inference_fn(prompt: str, **kw) -> str``
        used to ask the LLM to plan tool chains.
    """

    def __init__(self, tool_registry: ToolRegistry, inference_fn: Callable) -> None:
        self._registry = tool_registry
        self._infer = inference_fn

    # ── planning ────────────────────────────────────────────────────

    def plan_tool_chain(
        self,
        goal: str,
        available_tools: Optional[list[str]] = None,
    ) -> list[dict]:
        """Use the LLM to plan a sequence of tool calls for *goal*.

        Returns a list of step dicts::

            [
                {"tool": "name", "args": {...}, "depends_on": []},
                {"tool": "name", "args": {"query": "$step.0"}, "depends_on": [0]},
            ]

        Values containing ``$step.<N>`` are resolved at execution time to
        the output of step *N*.
        """
        if available_tools is None:
            available_tools = [t.name for t in self._registry.list_tools()]

        prompt = (
            "You are a tool-chaining planner.  Given a goal and a list of "
            "available tools, output ONLY a JSON array of steps.\n\n"
            "Each step is an object with keys:\n"
            '  "tool": tool name (string)\n'
            '  "args": arguments dict (values may use "$step.<N>" to reference '
            "an earlier step's output)\n"
            '  "depends_on": list of step indices this step depends on\n\n'
            f"Available tools: {json.dumps(available_tools)}\n"
            f"Goal: {goal}\n\n"
            "Respond with ONLY the JSON array, no other text."
        )

        raw = self._infer(prompt)
        return self._parse_plan(raw)

    # ── execution ───────────────────────────────────────────────────

    def execute_chain(self, chain: list[dict]) -> list[dict]:
        """Execute a planned tool chain, passing results between steps.

        Each entry in *chain* must have ``"tool"`` and ``"args"`` keys (and
        optionally ``"depends_on"``).  Argument values containing
        ``$step.<N>`` are resolved to the output of step *N* before
        execution.

        Returns a list of result dicts with ``tool``, ``success``,
        ``output``, and ``error`` keys.
        """
        results: list[dict] = []

        for idx, step in enumerate(chain):
            tool_name = step.get("tool", "")
            raw_args = step.get("args", {})

            # Validate tool exists before attempting execution.
            if not self._registry.get(tool_name):
                logger.warning("Tool '%s' not found at step %d — skipping.", tool_name, idx)
                results.append(
                    {"tool": tool_name, "success": False, "output": "", "error": f"Unknown tool: {tool_name}"}
                )
                continue

            # Resolve $step.N references in argument values.
            resolved_args = self._resolve_args(raw_args, results)

            logger.info("Executing step %d: %s(%s)", idx, tool_name, resolved_args)
            result = self.execute_single(tool_name, resolved_args)
            results.append(result)

            # If a step that others depend on fails, log but continue
            # (downstream steps will still see the error in $step.N).
            if not result["success"]:
                logger.warning("Step %d (%s) failed: %s", idx, tool_name, result.get("error"))

        return results

    def execute_single(self, tool_name: str, args: dict) -> dict:
        """Execute a single tool call and return a result dict.

        Returns
        -------
        dict
            ``{"tool": str, "success": bool, "output": str, "error": str | None}``
        """
        if not self._registry.get(tool_name):
            logger.error("Cannot execute unknown tool: %s", tool_name)
            return {"tool": tool_name, "success": False, "output": "", "error": f"Unknown tool: {tool_name}"}

        call = ToolCall(name=tool_name, arguments=args)
        try:
            result: ToolResult = self._registry.execute(call)
            return {
                "tool": result.tool_name,
                "success": result.success,
                "output": result.output,
                "error": result.error,
            }
        except Exception as exc:
            logger.exception("Unhandled error executing tool '%s'", tool_name)
            return {"tool": tool_name, "success": False, "output": "", "error": str(exc)}

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_args(args: dict[str, Any], results: list[dict]) -> dict[str, Any]:
        """Replace ``$step.<N>`` placeholders in *args* with prior results."""
        resolved: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                resolved[key] = _STEP_REF_PATTERN.sub(
                    lambda m: results[int(m.group(1))].get("output", "") if int(m.group(1)) < len(results) else m.group(0),
                    value,
                )
            else:
                resolved[key] = value
        return resolved

    @staticmethod
    def _parse_plan(raw: str) -> list[dict]:
        """Parse the LLM's JSON response into a list of step dicts."""
        # Strip markdown code fences if present.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            plan = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Failed to parse tool-chain plan: %s", raw[:200])
            return []

        if not isinstance(plan, list):
            logger.error("Tool-chain plan is not a list: %s", type(plan))
            return []

        # Normalise each step.
        normalised: list[dict] = []
        for step in plan:
            if not isinstance(step, dict) or "tool" not in step:
                continue
            normalised.append({
                "tool": step["tool"],
                "args": step.get("args", {}),
                "depends_on": step.get("depends_on", []),
            })
        return normalised
