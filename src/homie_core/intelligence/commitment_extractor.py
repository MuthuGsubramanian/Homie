from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

_COMMITMENT_PATTERNS = [
    re.compile(r"i(?:'ll| will| need to| have to|'m going to)\s+(.{5,80}?)(?:\s+by\s+(.{3,30}))?(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:remind me to|don't let me forget to)\s+(.{5,80})", re.IGNORECASE),
    re.compile(r"(?:todo|to-do|action item):\s*(.{5,120})", re.IGNORECASE),
]

@dataclass
class Commitment:
    text: str
    due_by: Optional[str] = None  # raw string like "tomorrow", "Friday", "end of week"
    source: str = "conversation"
    confidence: float = 0.7

def extract_commitments(text: str, source: str = "conversation") -> list[Commitment]:
    """Extract commitments from user text."""
    commitments = []
    for pattern in _COMMITMENT_PATTERNS:
        for match in pattern.finditer(text):
            task = match.group(1).strip()
            due = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else None
            if len(task) >= 5:  # skip very short matches
                commitments.append(Commitment(text=task, due_by=due, source=source))
    return commitments
