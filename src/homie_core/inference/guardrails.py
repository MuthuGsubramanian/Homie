"""Inference guardrails — detect and recover from model failures.

Handles blank/empty responses by retrying with adjusted parameters
or rephrased prompts. This compensates for base model limitations
where certain prompt patterns trigger degenerate output.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

# Patterns that commonly trigger blank output in the base model
_PROBLEMATIC_PATTERNS = [
    r"^(check|schedule|send|run|execute|deploy)\s+.+and\s+.+",  # Command + "and" + command
    r"^(what do i have|what.s on my).+(today|plate|agenda)",     # Daily agenda queries
]


def is_blank_response(response: str) -> bool:
    """Check if a response is effectively blank or degenerate."""
    if not response:
        return True
    stripped = response.strip()
    if len(stripped) < 10:
        return True
    # Check for repetitive degenerate output
    words = stripped.split()
    if len(words) > 5:
        unique = set(words)
        if len(unique) < len(words) * 0.3:  # Less than 30% unique words
            return True
    return False


def rephrase_for_retry(user_message: str) -> str:
    """Rephrase a problematic prompt to avoid blank output.

    Converts direct commands to conversational questions that the
    base model handles better.
    """
    msg = user_message.strip()

    # "Check X and Y" → "Can you help me with X? Also Y"
    if re.match(r"^(check|schedule|send|run)\s+", msg, re.IGNORECASE):
        return f"Can you help me with this: {msg}"

    # "What do I have on my plate" → "What should I focus on"
    if "on my plate" in msg.lower() or "my agenda" in msg.lower():
        return "Give me a summary of what I should focus on today based on my emails, projects, and tasks."

    # "What's on my schedule" → conversational form
    if "schedule" in msg.lower() and ("what" in msg.lower() or "show" in msg.lower()):
        return "Can you tell me about my upcoming schedule and tasks?"

    # Generic fallback: add conversational prefix
    return f"I need your help with something: {msg}. Please provide a detailed response."


def guarded_inference(
    inference_fn: Callable,
    messages: list[dict],
    options: dict | None = None,
    max_retries: int = 2,
) -> str:
    """Call inference with blank-response detection and retry.

    If the first attempt returns blank, retries with a rephrased
    user message and/or adjusted temperature.

    Args:
        inference_fn: Function that takes (messages, options) and returns response string
        messages: Chat messages list
        options: Model options (temperature, etc.)
        max_retries: Number of retry attempts

    Returns:
        The model response string (guaranteed non-blank if possible)
    """
    options = options or {}

    # First attempt
    response = inference_fn(messages=messages, options=options)

    if not is_blank_response(response):
        return response

    # Extract user message for rephrasing
    user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    if not user_msg:
        return response  # Can't rephrase without user message

    logger.info("Blank response detected, retrying with rephrase for: %s", user_msg[:60])

    for attempt in range(max_retries):
        # Rephrase the user message
        rephrased = rephrase_for_retry(user_msg)

        # Modify the messages with rephrased version
        retry_messages = []
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content") == user_msg:
                retry_messages.append({"role": "user", "content": rephrased})
            else:
                retry_messages.append(msg)

        # Adjust temperature slightly
        retry_options = dict(options)
        retry_options["temperature"] = min(0.9, options.get("temperature", 0.7) + 0.1 * (attempt + 1))

        response = inference_fn(messages=retry_messages, options=retry_options)

        if not is_blank_response(response):
            logger.info("Retry %d succeeded with rephrased prompt", attempt + 1)
            return response

    # All retries failed — return a helpful fallback
    logger.warning("All retries failed for: %s", user_msg[:60])
    return (
        "I'm having a moment processing that specific phrasing. "
        "Could you try rephrasing your request? For example, instead of "
        "a direct command, try asking it as a question like "
        "'Can you help me with...' or 'What should I focus on today?'"
    )
