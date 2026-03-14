import re


class PIIFilter:
    """Strips PII patterns from text before processing."""

    # Order matters: more specific patterns first, PHONE before SSN to avoid false matches
    _PATTERNS = [
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
        (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CARD]"),
        (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
        (re.compile(r"\b(?!(?:\d{3}[-.\s]?\d{3}))\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"), "[SSN]"),
    ]

    def filter(self, text: str) -> str:
        if not text:
            return text
        for pattern, replacement in self._PATTERNS:
            text = pattern.sub(replacement, text)
        return text
