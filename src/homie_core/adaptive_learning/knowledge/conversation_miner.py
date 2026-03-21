"""Conversation miner — extracts facts, decisions, and relationships from turns."""

import logging
import re
from typing import Callable, Optional

from ..storage import LearningStorage

logger = logging.getLogger(__name__)

# Quick extraction patterns (no LLM needed)
_FACT_PATTERNS = [
    (r"I\s+work\s+(?:at|for)\s+(\w[\w\s]*\w)", "works at {0}"),
    (r"I(?:'m|\s+am)\s+a\s+(\w[\w\s]*\w)", "is a {0}"),
    (r"I\s+prefer\s+(\w[\w\s]*?\w)\s+over\s+(\w[\w\s]*\w)", "prefers {0} over {1}"),
    (r"I\s+prefer\s+(\w[\w\s]*\w)", "prefers {0}"),
    (r"(?:I'm|I\s+am)\s+working\s+on\s+(?:the\s+)?(\w[\w\s]*\w?)(?:\s+project)?", "working on {0}"),
    (r"my\s+(?:main|primary)\s+(?:language|lang)\s+is\s+(\w+)", "primary language is {0}"),
    (r"I\s+use\s+(\w+)\s+(?:for|as)\s+(?:my\s+)?(\w[\w\s]*\w)", "uses {0} for {1}"),
]


class ConversationMiner:
    """Extracts structured knowledge from conversation turns."""

    def __init__(
        self,
        storage: LearningStorage,
        inference_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._storage = storage
        self._infer = inference_fn

    def extract_quick(self, text: str) -> list[str]:
        """Quick regex-based fact extraction (no LLM)."""
        if not text.strip():
            return []

        facts = []
        for pattern, template in _FACT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                fact = template.format(*[g.strip() for g in groups])
                facts.append(fact)
        return facts

    def process_turn(self, user_message: str, response: str) -> list[str]:
        """Process a conversation turn and extract knowledge."""
        facts = self.extract_quick(user_message)

        # Store extracted facts
        for fact in facts:
            domain = self._guess_domain(fact)
            self._storage.write_decision(fact, domain)

        return facts

    def _guess_domain(self, fact: str) -> str:
        """Simple domain classification for a fact."""
        fact_lower = fact.lower()
        if any(w in fact_lower for w in ["code", "python", "javascript", "git", "project", "programming"]):
            return "coding"
        if any(w in fact_lower for w in ["work", "job", "company", "team"]):
            return "work"
        return "general"
