"""Agentic Loop — multi-turn reasoning with tool use and self-correction.

The loop:
1. Build prompt (cognitive architecture) + tool descriptions
2. Generate response
3. Check for tool calls in output
4. If tool calls found: execute tools, append results, re-generate
5. If a tool fails: give the model a chance to self-correct
6. Repeat until no more tool calls or max iterations reached
7. Return final response (with tool call markers stripped)

This transforms Homie from a one-shot chatbot into an agent that can
reason, act, observe results, self-correct, and reason again.
"""
from __future__ import annotations

import re
from typing import Iterator, Optional

from homie_core.brain.tool_registry import ToolRegistry, ToolResult, parse_tool_calls


# Strip tool call markers from final output shown to user
_TOOL_MARKER_PATTERN = re.compile(r"<tool>.*?</tool>", re.DOTALL)
_JSON_TOOL_MARKER = re.compile(r'\{"tool"\s*:.*?\}', re.DOTALL)
_ACTION_MARKER = re.compile(r'Action:\s*\w+\s*\([^)]*\)', re.IGNORECASE)
_MARKDOWN_TOOL_MARKER = re.compile(r'```(?:tool|tool_code|json)?\s*\n\s*\{"name".*?\}\s*\n```', re.DOTALL)

# Maximum tool-use iterations to prevent infinite loops
_MAX_ITERATIONS = 7


def _strip_tool_markers(text: str) -> str:
    """Remove tool call syntax from text shown to user."""
    text = _TOOL_MARKER_PATTERN.sub("", text)
    text = _JSON_TOOL_MARKER.sub("", text)
    text = _ACTION_MARKER.sub("", text)
    text = _MARKDOWN_TOOL_MARKER.sub("", text)
    # Clean up extra whitespace left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_tool_results(results: list[ToolResult]) -> tuple[str, bool]:
    """Format tool results as context for the next generation round.

    Returns (formatted_text, has_errors).
    """
    parts = []
    has_errors = False
    for r in results:
        if r.success:
            parts.append(f"[Tool: {r.tool_name}] Result: {r.output}")
        else:
            parts.append(f"[Tool: {r.tool_name}] Error: {r.error}")
            has_errors = True
    return "\n".join(parts), has_errors


class AgenticLoop:
    """Multi-turn agentic reasoning loop with tool use and self-correction.

    Wraps a model engine and tool registry. Each call to process()
    may trigger multiple generation rounds if the model invokes tools.
    Includes self-correction: when a tool fails, the model gets guidance
    to try a different approach.
    """

    def __init__(
        self,
        model_engine,
        tool_registry: ToolRegistry,
        max_iterations: int = _MAX_ITERATIONS,
    ):
        self._engine = model_engine
        self._tools = tool_registry
        self._max_iterations = max_iterations

    def process(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Run the agentic loop — blocking mode.

        Returns the final response with tool markers stripped.
        """
        current_prompt = prompt
        all_text_parts = []
        consecutive_errors = 0

        for iteration in range(self._max_iterations):
            # Generate
            response = self._engine.generate(
                current_prompt, max_tokens=max_tokens, temperature=temperature,
            )

            # Check for tool calls
            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                # No tool calls — this is the final response
                all_text_parts.append(response)
                break

            # Execute tools
            results = [self._tools.execute(call) for call in tool_calls]

            # Collect the text before/between tool calls
            clean_text = _strip_tool_markers(response)
            if clean_text:
                all_text_parts.append(clean_text)

            # Format results and check for errors
            tool_output, has_errors = _format_tool_results(results)

            # Self-correction: track consecutive errors
            if has_errors:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    all_text_parts.append(
                        "I encountered repeated tool errors. Let me answer based on what I know."
                    )
                    break
                error_guidance = (
                    "\nThe tool call failed. Try a different approach — use different "
                    "parameters, a different tool, or answer without tools if possible."
                )
                tool_output += error_guidance
            else:
                consecutive_errors = 0

            # Append tool results and re-prompt
            current_prompt = (
                f"{current_prompt}\n\nAssistant: {response}\n\n{tool_output}\n\n"
                f"Continue your response based on the tool results above.\nAssistant:"
            )
        else:
            # Max iterations reached
            all_text_parts.append(
                "(Reached maximum reasoning steps. Here's what I have so far.)"
            )

        final = "\n".join(all_text_parts)
        return _strip_tool_markers(final)

    def process_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Run the agentic loop — streaming mode.

        Yields tokens in real-time. When a tool call is detected in accumulated
        output, pauses streaming, executes tools, then continues generation
        with a new streaming round.

        Fix: tokens buffered before tool detection are properly yielded
        after tool execution, and self-correction is applied on errors.
        """
        current_prompt = prompt
        consecutive_errors = 0

        for iteration in range(self._max_iterations):
            # Stream tokens, accumulating for tool call detection
            accumulated = []
            yielded_count = 0  # Track how many tokens we've already yielded
            tool_detected = False

            for token in self._engine.stream(
                current_prompt, max_tokens=max_tokens, temperature=temperature,
            ):
                accumulated.append(token)

                # Check if we've seen a complete tool call
                full_text = "".join(accumulated)
                if "<tool>" in full_text and "</tool>" in full_text:
                    tool_detected = True
                    break

                # Only yield tokens that are clearly not part of a tool call
                if "<tool>" not in full_text:
                    yield token
                    yielded_count = len(accumulated)

            full_response = "".join(accumulated)

            if not tool_detected:
                # No tool calls — yield any tokens that were buffered
                # (tokens accumulated after last yield, before we detected no-tool)
                if yielded_count < len(accumulated):
                    remaining_text = "".join(accumulated[yielded_count:])
                    clean = _strip_tool_markers(remaining_text)
                    if clean:
                        yield clean
                break

            # Parse and execute tool calls
            tool_calls = parse_tool_calls(full_response)
            if not tool_calls:
                # False positive — yield the buffered text
                if yielded_count < len(accumulated):
                    remaining_text = "".join(accumulated[yielded_count:])
                    clean = _strip_tool_markers(remaining_text)
                    if clean:
                        yield clean
                break

            results = [self._tools.execute(call) for call in tool_calls]
            tool_output, has_errors = _format_tool_results(results)

            # Self-correction on errors
            if has_errors:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    yield "\nI encountered repeated tool errors. Let me answer based on what I know.\n"
                    break
                tool_output += (
                    "\nThe tool call failed. Try a different approach or answer without tools."
                )
            else:
                consecutive_errors = 0

            # Yield a brief indicator that tools executed
            yield "\n"

            # Build continuation prompt
            current_prompt = (
                f"{current_prompt}\n\nAssistant: {full_response}\n\n{tool_output}\n\n"
                f"Continue your response based on the tool results above.\nAssistant:"
            )
