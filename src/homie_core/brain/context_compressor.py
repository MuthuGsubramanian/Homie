"""Context Compressor — automatic conversation compression for token budgets.

Long conversations blow past token limits and degrade response quality.
This module compresses conversation history while preserving the critical
context Homie needs: the opening exchange (establishes intent), the recent
exchange (working context), and a distilled summary of everything in between.

Algorithm:
1. Check total character count against configurable threshold
2. Identify compression zone (middle messages between protected head/tail)
3. Adjust boundaries so tool-call / tool-result pairs are never orphaned
4. Run lightweight extractive summarization on the middle section
5. Replace middle messages with a single [COMPRESSED] system message

No model calls required — summarization is purely extractive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# Patterns for detecting tool interactions in message content
_TOOL_CALL_PATTERN = re.compile(r"<tool>\s*\w+\s*\(.*?\)\s*</tool>", re.DOTALL)
_TOOL_RESULT_PATTERN = re.compile(r"\[Tool:\s*\w+\]\s*(?:Result|Error):", re.DOTALL)

# Sentence boundary splitter (handles ., !, ? followed by space or end)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Keywords that signal information-bearing user statements
_INFO_KEYWORDS = re.compile(
    r"\b(?:how|why|what|when|where|which|can you|could you|please|should|"
    r"explain|show|find|create|fix|update|delete|error|issue|problem|"
    r"want|need|help|configure|setup|install|run|build|deploy)\b",
    re.IGNORECASE,
)


@dataclass
class CompressionStats:
    """Diagnostics for a single compression pass."""

    original_messages: int = 0
    compressed_messages: int = 0
    original_chars: int = 0
    compressed_chars: int = 0
    middle_messages_summarized: int = 0


class ContextCompressor:
    """Compresses conversation history to stay within token budgets.

    Preserves the first N and last N messages verbatim, collapses the
    middle section into an extractive summary, and never splits
    tool-call / tool-result pairs.

    Usage::

        compressor = ContextCompressor(threshold_chars=10000)
        conversation = working_memory.get_conversation()

        if compressor.needs_compression(conversation):
            conversation = compressor.compress(conversation)
    """

    def __init__(
        self,
        threshold_chars: int = 10000,
        protect_first_n: int = 2,
        protect_last_n: int = 3,
        summary_target_chars: int = 800,
    ):
        """Initialise the compressor.

        Args:
            threshold_chars: Total character count that triggers compression.
            protect_first_n: Number of messages to keep at the start.
            protect_last_n: Number of messages to keep at the end.
            summary_target_chars: Target length for the compressed summary.
        """
        self._threshold = threshold_chars
        self._protect_first = protect_first_n
        self._protect_last = protect_last_n
        self._summary_target = summary_target_chars
        self._compression_count: int = 0
        self._last_stats: CompressionStats | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def compression_count(self) -> int:
        """Total number of compressions performed by this instance."""
        return self._compression_count

    @property
    def last_stats(self) -> CompressionStats | None:
        """Stats from the most recent compression, or ``None``."""
        return self._last_stats

    def needs_compression(self, conversation: list[dict]) -> bool:
        """Return ``True`` if the conversation exceeds the character threshold."""
        total = sum(len(msg.get("content", "")) for msg in conversation)
        return total > self._threshold

    def compress(self, conversation: list[dict]) -> list[dict]:
        """Compress the conversation, preserving head/tail and tool pairs.

        Returns a new list — the original is not mutated.
        """
        n = len(conversation)
        original_chars = sum(len(m.get("content", "")) for m in conversation)

        # Nothing to compress if the conversation is short enough to be
        # fully covered by the protected zones.
        min_protected = self._protect_first + self._protect_last
        if n <= min_protected:
            return list(conversation)

        # Raw compression boundaries (indices into conversation)
        raw_start = self._protect_first
        raw_end = n - self._protect_last  # exclusive upper bound

        if raw_start >= raw_end:
            return list(conversation)

        # Adjust boundaries so tool-call/result pairs stay together
        comp_start, comp_end = self._detect_tool_boundaries(
            conversation, raw_start, raw_end,
        )

        # If adjustment collapsed the zone, nothing to compress
        if comp_start >= comp_end:
            return list(conversation)

        middle = conversation[comp_start:comp_end]
        summary_text = self._extractive_summarize(middle, self._summary_target)

        summary_message: dict = {
            "role": "system",
            "content": (
                f"[COMPRESSED] The following is a summary of "
                f"{len(middle)} earlier messages:\n{summary_text}"
            ),
        }

        result = (
            list(conversation[:comp_start])
            + [summary_message]
            + list(conversation[comp_end:])
        )

        # Diagnostics
        compressed_chars = sum(len(m.get("content", "")) for m in result)
        self._last_stats = CompressionStats(
            original_messages=n,
            compressed_messages=len(result),
            original_chars=original_chars,
            compressed_chars=compressed_chars,
            middle_messages_summarized=len(middle),
        )
        self._compression_count += 1

        return result

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    def _detect_tool_boundaries(
        self,
        conversation: list[dict],
        start: int,
        end: int,
    ) -> tuple[int, int]:
        """Adjust *start* and *end* so tool-call/result pairs are not split.

        A tool-call message (assistant with ``<tool>...</tool>``) is always
        followed by a tool-result message (``[Tool: ...] Result: ...``).
        If the boundary falls between the two, shift it to keep the pair
        together in the same zone.

        Args:
            conversation: Full conversation list.
            start: Proposed first index of the compression zone.
            end: Proposed exclusive upper-bound of the compression zone.

        Returns:
            Adjusted ``(start, end)`` tuple.
        """
        n = len(conversation)

        # --- Fix the start boundary ---
        # If the message just before start is a tool call, its result
        # (at index start) belongs with it — shrink the zone.
        if start > 0 and self._is_tool_call(conversation[start - 1]):
            start = min(start + 1, n)

        # If the first message inside the zone is a tool result, its
        # call (at start - 1) is protected — pull the result out too.
        if start < n and self._is_tool_result(conversation[start]):
            start = min(start + 1, n)

        # --- Fix the end boundary ---
        # If the last message in the zone (end - 1) is a tool call,
        # the result (at end) is in the protected tail — expand zone.
        if end > 0 and end < n and self._is_tool_call(conversation[end - 1]):
            end = max(end - 1, 0)

        # If the first protected tail message is a tool result, its
        # call is the last zone message — pull the call into protection.
        if end < n and self._is_tool_result(conversation[end]):
            end = max(end - 1, 0)

        return start, end

    # ------------------------------------------------------------------
    # Extractive summarization
    # ------------------------------------------------------------------

    def _extractive_summarize(
        self,
        messages: list[dict],
        target_chars: int,
    ) -> str:
        """Build a summary by extracting key sentences from *messages*.

        Strategy:
        - **User messages**: keep sentences that contain information-bearing
          keywords (questions, requests, problem descriptions).
        - **Assistant messages**: keep the first and last sentence
          (typically conclusion / recommendation).
        - **Tool-result messages**: prioritised — they record action outcomes.

        The result is truncated to *target_chars*.
        """
        extracted: list[tuple[float, str]] = []  # (priority, sentence)

        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            role = msg.get("role", "")
            has_tool_result = bool(_TOOL_RESULT_PATTERN.search(content))

            # Base priority: tool results > user > assistant > system
            if has_tool_result:
                base_priority = 3.0
            elif role == "user":
                base_priority = 2.0
            elif role == "assistant":
                base_priority = 1.0
            else:
                base_priority = 0.5

            sentences = _SENTENCE_SPLIT.split(content.strip())
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                continue

            if role == "user":
                # Keep information-bearing sentences
                for sent in sentences:
                    priority = base_priority
                    if _INFO_KEYWORDS.search(sent):
                        priority += 1.0
                    extracted.append((priority, sent))

            elif role == "assistant" or has_tool_result:
                # First sentence (introduction / action) and last (conclusion)
                extracted.append((base_priority + 0.5, sentences[0]))
                if len(sentences) > 1:
                    extracted.append((base_priority + 0.5, sentences[-1]))
                # For tool results, also grab the result line itself
                if has_tool_result:
                    for sent in sentences:
                        if _TOOL_RESULT_PATTERN.search(sent):
                            extracted.append((base_priority + 1.0, sent))

            else:
                # System messages — low priority, first sentence only
                extracted.append((base_priority, sentences[0]))

        # Sort by descending priority, then assemble
        extracted.sort(key=lambda x: x[0], reverse=True)

        parts: list[str] = []
        current_len = 0
        seen: set[str] = set()

        for _priority, sentence in extracted:
            # Deduplicate identical sentences
            normalised = sentence.lower().strip()
            if normalised in seen:
                continue
            seen.add(normalised)

            addition = len(sentence) + 1  # +1 for the space/newline
            if current_len + addition > target_chars:
                # Try to fit a truncated version
                remaining = target_chars - current_len
                if remaining > 40:
                    parts.append(sentence[: remaining - 3] + "...")
                break
            parts.append(sentence)
            current_len += addition

        return " ".join(parts) if parts else "(no extractable content)"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tool_call(message: dict) -> bool:
        """Return ``True`` if *message* contains a tool invocation."""
        content = message.get("content", "")
        return bool(_TOOL_CALL_PATTERN.search(content))

    @staticmethod
    def _is_tool_result(message: dict) -> bool:
        """Return ``True`` if *message* contains tool output."""
        content = message.get("content", "")
        return bool(_TOOL_RESULT_PATTERN.search(content))
