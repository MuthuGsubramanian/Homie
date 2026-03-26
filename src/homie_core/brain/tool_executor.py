"""Tool executor — bridges action templates to real tool execution.

When Homie detects an intent that maps to an action template,
this executor runs the tool sequence and returns formatted results.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from homie_core.brain.action_templates import (
    ActionTemplate, detect_intent, get_action_template,
)
from homie_core.brain.tool_registry import ToolCall

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes action template tool sequences against a tool registry."""

    def __init__(self, tool_registry=None):
        self._registry = tool_registry

    def can_execute(self, user_message: str) -> bool:
        """Check if we can execute an action for this message."""
        if not self._registry:
            return False
        intent = detect_intent(user_message)
        if not intent:
            return False
        template = get_action_template(intent)
        if not template:
            return False
        # Check all required tools are registered
        for tool_name in template.required_tools:
            if not self._registry.has_tool(tool_name):
                return False
        return True

    def execute(self, user_message: str, context: dict | None = None) -> Optional[dict]:
        """Execute the action template for a user message.

        Returns:
            dict with keys:
                - "intent": the detected intent
                - "template": the action template used
                - "results": list of tool execution results
                - "formatted": human-readable formatted output
                - "needs_confirmation": whether user should confirm before proceeding
            None if no matching intent or tools unavailable.
        """
        intent = detect_intent(user_message)
        if not intent:
            return None

        template = get_action_template(intent)
        if not template:
            return None

        if not self._registry:
            return {
                "intent": intent,
                "template": template,
                "results": [],
                "formatted": f"I would {template.description}, but tool execution is not available in this mode.",
                "needs_confirmation": False,
            }

        # Extract parameters from user message
        params = self._extract_params(user_message, intent)

        # Check if confirmation needed before executing
        if template.confirmation_required:
            return {
                "intent": intent,
                "template": template,
                "results": [],
                "formatted": f"I'm ready to {template.description}. Shall I proceed?",
                "needs_confirmation": True,
                "params": params,
            }

        # Execute tool sequence
        results = self._run_tool_sequence(template, params)

        # Format results
        formatted = self._format_results(template, results)

        return {
            "intent": intent,
            "template": template,
            "results": results,
            "formatted": formatted,
            "needs_confirmation": False,
        }

    def execute_confirmed(self, intent: str, params: dict) -> dict:
        """Execute a previously-confirmed action."""
        template = get_action_template(intent)
        if not template or not self._registry:
            return {"results": [], "formatted": "Could not execute — template or tools unavailable."}

        results = self._run_tool_sequence(template, params)
        return {"results": results, "formatted": self._format_results(template, results)}

    def _run_tool_sequence(self, template: ActionTemplate, params: dict) -> list[dict]:
        """Run the tool sequence from a template, substituting params."""
        results = []
        for step in template.tool_sequence:
            tool_name = step["tool"]
            args = dict(step.get("args", {}))

            # Substitute parameters
            for key, value in args.items():
                if isinstance(value, str) and "{" in value:
                    for param_key, param_val in params.items():
                        value = value.replace(f"{{{param_key}}}", str(param_val))
                    args[key] = value

            try:
                if self._registry.has_tool(tool_name):
                    call = ToolCall(name=tool_name, arguments=args)
                    tool_result = self._registry.execute(call)
                    results.append({
                        "tool": tool_name,
                        "result": tool_result.output,
                        "success": tool_result.success,
                    })
                else:
                    results.append({
                        "tool": tool_name,
                        "result": f"Tool {tool_name} not available",
                        "success": False,
                    })
            except Exception as exc:
                logger.warning("Tool %s execution failed: %s", tool_name, exc)
                results.append({"tool": tool_name, "result": str(exc), "success": False})

        return results

    def _extract_params(self, user_message: str, intent: str) -> dict:
        """Extract relevant parameters from the user message."""
        params = {}
        msg = user_message.strip()

        # Email-related params
        if intent in ("search_email", "draft_email"):
            # Extract "to" recipient: "email to John" -> John
            to_match = re.search(r"(?:to|for)\s+(\w+(?:\s+\w+)?)(?=\s+about|\s*$|\.)", msg, re.IGNORECASE)
            if to_match:
                params["to"] = to_match.group(1)

            # Extract "about" subject
            about_match = re.search(r"about\s+(.+?)(?:\.|$)", msg, re.IGNORECASE)
            if about_match:
                params["subject"] = about_match.group(1).strip()
                params["query"] = about_match.group(1).strip()

        # Remember/recall params
        if intent == "remember_fact":
            # "Remember that I prefer dark mode" -> "I prefer dark mode"
            fact_match = re.search(r"remember\s+(?:that\s+)?(.+)", msg, re.IGNORECASE)
            if fact_match:
                params["fact"] = fact_match.group(1).strip()

        if intent == "recall_info":
            # "What do you know about X" -> X
            query_match = re.search(r"(?:know\s+about|remember|recall)\s+(.+)", msg, re.IGNORECASE)
            if query_match:
                params["query"] = query_match.group(1).strip()

        # File params
        if intent in ("list_files", "read_file"):
            # Extract file path or pattern
            path_match = re.search(r"(?:file|path)\s+[\"']?([^\s\"']+)", msg, re.IGNORECASE)
            if path_match:
                params["path"] = path_match.group(1)
                params["pattern"] = path_match.group(1)

        return params

    def _format_results(self, template: ActionTemplate, results: list[dict]) -> str:
        """Format tool execution results for display."""
        if not results:
            return "No results."

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        parts = []
        for r in successful:
            result_text = str(r["result"])[:1000]  # Cap display length
            parts.append(result_text)

        if failed:
            parts.append(f"Note: {len(failed)} tool(s) encountered issues.")

        combined = "\n".join(parts) if parts else "No results."

        if template.response_template and "{result}" in template.response_template:
            return template.response_template.replace("{result}", combined)
        return combined
