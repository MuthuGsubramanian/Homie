"""WorkflowEngine — executes complex workflows with conditional logic and error handling.

Builds on :class:`ToolOrchestrator` to provide higher-level workflow
primitives: conditional steps, retry / skip / stop failure policies, and
LLM-driven workflow generation from natural-language goals.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .tool_orchestrator import ToolOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """A single step in a workflow.

    Attributes
    ----------
    tool:
        Name of the tool to invoke.
    args:
        Keyword arguments forwarded to the tool.
    condition:
        Optional Python-style expression evaluated against prior results.
        The step runs only when the condition is truthy (or absent).
    on_failure:
        Policy when the step fails: ``"stop"`` (default), ``"skip"``, or
        ``"retry"`` (retries once).
    max_retries:
        Number of retry attempts when *on_failure* is ``"retry"``.
    result:
        Populated after execution with the result dict.
    """

    tool: str
    args: dict = field(default_factory=dict)
    condition: Optional[str] = None
    on_failure: str = "stop"        # stop | skip | retry
    max_retries: int = 1
    result: Optional[dict] = None


class WorkflowEngine:
    """Executes complex workflows with conditional logic and error handling.

    Parameters
    ----------
    tool_orchestrator:
        The :class:`ToolOrchestrator` used for individual tool execution.
    """

    def __init__(self, tool_orchestrator: ToolOrchestrator) -> None:
        self._orchestrator = tool_orchestrator

    # ── execution ───────────────────────────────────────────────────

    def run_workflow(self, steps: list[WorkflowStep]) -> dict:
        """Execute a workflow, returning a summary dict.

        Returns
        -------
        dict
            ``{"status": "completed"|"failed"|"partial", "steps": [...],
            "completed": int, "failed": int, "skipped": int}``
        """
        completed = 0
        failed = 0
        skipped = 0
        results: list[dict] = []

        for idx, step in enumerate(steps):
            # ── condition check ──────────────────────────────────
            if step.condition and not self._evaluate_condition(step.condition, results):
                logger.info("Step %d (%s) skipped — condition not met: %s", idx, step.tool, step.condition)
                step.result = {"tool": step.tool, "success": True, "output": "", "error": None, "skipped": True}
                results.append(step.result)
                skipped += 1
                continue

            # ── execute (with optional retry) ────────────────────
            attempts = 1 + (step.max_retries if step.on_failure == "retry" else 0)
            result: Optional[dict] = None

            for attempt in range(attempts):
                logger.info("Step %d (%s) — attempt %d/%d", idx, step.tool, attempt + 1, attempts)
                result = self._orchestrator.execute_single(step.tool, step.args)
                if result["success"]:
                    break
                logger.warning("Step %d (%s) attempt %d failed: %s", idx, step.tool, attempt + 1, result.get("error"))

            assert result is not None  # at least one attempt always runs
            step.result = result
            results.append(result)

            if result["success"]:
                completed += 1
            else:
                failed += 1
                if step.on_failure == "stop":
                    logger.error("Workflow stopped at step %d (%s).", idx, step.tool)
                    # Mark remaining steps as skipped
                    for remaining in steps[idx + 1:]:
                        remaining.result = {
                            "tool": remaining.tool, "success": False,
                            "output": "", "error": "Workflow stopped", "skipped": True,
                        }
                        results.append(remaining.result)
                        skipped += 1
                    break
                elif step.on_failure == "skip":
                    logger.info("Step %d (%s) failed — skipping per policy.", idx, step.tool)
                    # Treat as skipped after failure — workflow continues.

        status = "completed" if failed == 0 else ("failed" if completed == 0 else "partial")
        return {
            "status": status,
            "steps": results,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
        }

    # ── LLM-driven workflow creation ────────────────────────────────

    def create_workflow_from_goal(self, goal: str) -> list[WorkflowStep]:
        """Use the LLM to create a workflow from a natural-language goal.

        Delegates to the orchestrator's inference function to produce a
        list of :class:`WorkflowStep` objects.
        """
        available = [t.name for t in self._orchestrator._registry.list_tools()]

        prompt = (
            "You are a workflow planner.  Given a goal and available tools, "
            "output ONLY a JSON array of workflow steps.\n\n"
            "Each step object has keys:\n"
            '  "tool": tool name (string)\n'
            '  "args": arguments dict\n'
            '  "condition": optional condition string (null if none)\n'
            '  "on_failure": "stop", "skip", or "retry"\n\n'
            f"Available tools: {json.dumps(available)}\n"
            f"Goal: {goal}\n\n"
            "Respond with ONLY the JSON array."
        )

        raw = self._orchestrator._infer(prompt)
        return self._parse_workflow(raw)

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _evaluate_condition(condition: str, prior_results: list[dict]) -> bool:
        """Safely evaluate a condition string against prior step results.

        The condition can reference ``results`` (the list of prior result
        dicts) and ``last`` (the most recent result).  Only a restricted
        set of builtins is available to prevent arbitrary code execution.
        """
        if not condition:
            return True

        safe_globals: dict[str, Any] = {"__builtins__": {}}
        safe_locals: dict[str, Any] = {
            "results": prior_results,
            "last": prior_results[-1] if prior_results else {},
            "len": len,
            "any": any,
            "all": all,
            "True": True,
            "False": False,
        }

        try:
            return bool(eval(condition, safe_globals, safe_locals))  # noqa: S307
        except Exception:
            logger.warning("Condition evaluation failed for '%s' — treating as False.", condition)
            return False

    @staticmethod
    def _parse_workflow(raw: str) -> list[WorkflowStep]:
        """Parse the LLM's JSON response into a list of WorkflowStep."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Failed to parse workflow plan: %s", raw[:200])
            return []

        if not isinstance(data, list):
            return []

        steps: list[WorkflowStep] = []
        for entry in data:
            if not isinstance(entry, dict) or "tool" not in entry:
                continue
            steps.append(WorkflowStep(
                tool=entry["tool"],
                args=entry.get("args", {}),
                condition=entry.get("condition"),
                on_failure=entry.get("on_failure", "stop"),
            ))
        return steps
