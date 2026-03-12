"""Prompt injection detection for external content entering the AI pipeline.

Screens clipboard pastes, file contents, screen captures (OCR text), and
user-supplied context for known injection patterns before they reach the
system prompt or tool inputs.

Design goals:
    - Zero external dependencies (stdlib ``re`` + ``unicodedata`` only).
    - All regex patterns compiled once at module level.
    - Deterministic threat scoring with clear category labels.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

THREAT_LEVELS = ("none", "low", "medium", "high", "critical")


@dataclass
class InjectionResult:
    """Outcome of an injection scan.

    Attributes:
        is_suspicious: ``True`` when at least one heuristic fires.
        threat_level: One of ``"none"``, ``"low"``, ``"medium"``,
            ``"high"``, or ``"critical"``.
        categories: Deduplicated list of matched category names.
        details: Human-readable descriptions of each matched pattern.
    """

    is_suspicious: bool = False
    threat_level: str = "none"
    categories: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
# Each entry is (compiled_regex, category, description, weight).
# ``weight`` drives the threat score: 1 = low-confidence, 2 = medium,
# 3 = high, 5 = obvious/critical.

_FLAGS = re.IGNORECASE | re.MULTILINE

_PATTERNS: List[Tuple[re.Pattern, str, str, int]] = []


def _p(pattern: str, category: str, description: str, weight: int) -> None:
    """Compile *pattern* and append it to the global pattern list."""
    _PATTERNS.append((re.compile(pattern, _FLAGS), category, description, weight))


# -- 1. INSTRUCTION_OVERRIDE -----------------------------------------------
_p(r"ignore\s+(all\s+)?previous\s+instructions",
   "INSTRUCTION_OVERRIDE", "Attempt to override prior instructions", 5)
_p(r"forget\s+(all\s+)?(your\s+)?instructions",
   "INSTRUCTION_OVERRIDE", "Attempt to erase system instructions", 5)
_p(r"system\s+prompt\s+override",
   "INSTRUCTION_OVERRIDE", "Explicit system-prompt override phrase", 5)
_p(r"(?<!\w)you\s+are\s+now\b",
   "INSTRUCTION_OVERRIDE", "Identity reassignment ('you are now')", 4)
_p(r"new\s+instructions\s*:",
   "INSTRUCTION_OVERRIDE", "Injected instruction block header", 5)
_p(r"disregard\s+(the\s+)?(above|previous|prior)",
   "INSTRUCTION_OVERRIDE", "Disregard-above directive", 5)
_p(r"do\s+not\s+follow\s+(any\s+)?(previous|prior|above)\s+(instructions|rules|guidelines)",
   "INSTRUCTION_OVERRIDE", "Instruction negation directive", 4)
_p(r"override\s+(all\s+)?(safety|content|security)\s+(filters?|policies|rules)",
   "INSTRUCTION_OVERRIDE", "Safety-filter override attempt", 5)

# -- 2. ROLE_HIJACK ---------------------------------------------------------
_p(r"(?<!\w)you\s+are\s+a\s+\w+",
   "ROLE_HIJACK", "Role reassignment ('you are a ...')", 2)
_p(r"pretend\s+(to\s+be|you\s+are)",
   "ROLE_HIJACK", "Pretend-to-be directive", 3)
_p(r"act\s+as\s+if\s+you\s+are",
   "ROLE_HIJACK", "Act-as-if directive", 3)
_p(r"roleplay\s+as",
   "ROLE_HIJACK", "Roleplay directive", 3)
_p(r"your\s+new\s+role\s+is",
   "ROLE_HIJACK", "Explicit new-role assignment", 4)
_p(r"from\s+now\s+on\s+you\s+(are|will|must|should)",
   "ROLE_HIJACK", "Temporal role reassignment", 3)

# -- 3. DATA_EXFILTRATION ---------------------------------------------------
_p(r"repeat\s+(your\s+)?(system\s+)?prompt",
   "DATA_EXFILTRATION", "System-prompt exfiltration request", 4)
_p(r"show\s+me\s+your\s+instructions",
   "DATA_EXFILTRATION", "Instruction disclosure request", 4)
_p(r"what\s+are\s+your\s+(rules|instructions|guidelines)",
   "DATA_EXFILTRATION", "Rules disclosure request", 3)
_p(r"print\s+your\s+(prompt|instructions|rules)",
   "DATA_EXFILTRATION", "Print-prompt directive", 4)
_p(r"output\s+your\s+(configuration|config|system\s+prompt|instructions)",
   "DATA_EXFILTRATION", "Configuration exfiltration request", 4)
_p(r"reveal\s+(your\s+)?(system|hidden|secret)\s+(prompt|instructions|message)",
   "DATA_EXFILTRATION", "Hidden prompt reveal attempt", 5)
_p(r"display\s+(the\s+)?(system|initial)\s+(prompt|message|instructions)",
   "DATA_EXFILTRATION", "System message display request", 4)

# -- 4. COMMAND_INJECTION ---------------------------------------------------
_p(r"(?:curl|wget)\s+.*(?:api[_-]?key|token|secret|password|auth)",
   "COMMAND_INJECTION", "HTTP request with credential parameter", 5)
_p(r"cat\s+/etc/(?:passwd|shadow)",
   "COMMAND_INJECTION", "Sensitive system file read", 5)
_p(r"rm\s+-rf\s+/",
   "COMMAND_INJECTION", "Destructive recursive delete at root", 5)
_p(r"(?:cat|less|more|head|tail|type)\s+.*\.env\b",
   "COMMAND_INJECTION", "Environment file read attempt", 4)
_p(r"\beval\s*\(",
   "COMMAND_INJECTION", "Dynamic code evaluation (eval)", 4)
_p(r"\bexec\s*\(",
   "COMMAND_INJECTION", "Dynamic code execution (exec)", 4)
_p(r"\bos\.system\s*\(",
   "COMMAND_INJECTION", "os.system call injection", 4)
_p(r"\bsubprocess\.\w+\s*\(",
   "COMMAND_INJECTION", "subprocess call injection", 3)
_p(r"(?:chmod|chown)\s+.*(?:777|\\+s)",
   "COMMAND_INJECTION", "Dangerous permission change", 4)
_p(r"(?:nc|ncat|netcat)\s+-[lp]",
   "COMMAND_INJECTION", "Netcat listener / reverse shell", 5)

# -- 5. UNICODE_TRICKS ------------------------------------------------------
# Detected via character-inspection heuristics rather than regex on the
# literal codepoints (see ``_check_unicode_tricks``).

# -- 6. ENCODING_ATTACKS ----------------------------------------------------
_p(r"aWdub3Jl",
   "ENCODING_ATTACK", "Base64-encoded 'ignore' detected", 3)
_p(r"(?:aW5zdHJ1Y3Rpb24|c3lzdGVtIHByb21wdA|Zm9yZ2V0)",
   "ENCODING_ATTACK", "Base64-encoded injection keyword detected", 3)
# Excessive \uXXXX or &#xXXXX; escapes in a short span
_p(r"(?:\\u[0-9a-fA-F]{4}){4,}",
   "ENCODING_ATTACK", "Excessive Unicode escape sequences", 2)
_p(r"(?:&#x[0-9a-fA-F]{2,6};){4,}",
   "ENCODING_ATTACK", "Excessive HTML Unicode escapes", 2)

# -- 7. CSS_DISPLAY_TRICKS --------------------------------------------------
_p(r"display\s*:\s*none",
   "CSS_DISPLAY_TRICK", "CSS hidden element (display:none)", 3)
_p(r"visibility\s*:\s*hidden",
   "CSS_DISPLAY_TRICK", "CSS hidden element (visibility:hidden)", 3)
_p(r"font-size\s*:\s*0",
   "CSS_DISPLAY_TRICK", "CSS zero font-size trick", 3)
_p(r"opacity\s*:\s*0(?:[;\s}]|$)",
   "CSS_DISPLAY_TRICK", "CSS fully-transparent element", 3)
_p(r"position\s*:\s*absolute\s*;[^}]*(?:left|top)\s*:\s*-\d{4,}",
   "CSS_DISPLAY_TRICK", "CSS off-screen positioning", 2)
_p(r"color\s*:\s*(?:transparent|rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*0\s*\))",
   "CSS_DISPLAY_TRICK", "CSS transparent text trick", 3)

# ---------------------------------------------------------------------------
# Unicode anomaly codepoint sets
# ---------------------------------------------------------------------------

_SUSPICIOUS_CODEPOINTS = frozenset({
    "\u200b",   # zero-width space
    "\u200c",   # zero-width non-joiner
    "\u200d",   # zero-width joiner
    "\u200e",   # LTR mark
    "\u200f",   # RTL mark
    "\u2060",   # word joiner
    "\u2061",   # function application
    "\u2062",   # invisible times
    "\u2063",   # invisible separator
    "\u2064",   # invisible plus
    "\ufeff",   # BOM / zero-width no-break space
    "\u202a",   # LTR embedding
    "\u202b",   # RTL embedding
    "\u202c",   # pop directional formatting
    "\u202d",   # LTR override
    "\u202e",   # RTL override
    "\u2066",   # LTR isolate
    "\u2067",   # RTL isolate
    "\u2068",   # first strong isolate
    "\u2069",   # pop directional isolate
    "\u00ad",   # soft hyphen
    "\ufff9",   # interlinear annotation anchor
    "\ufffa",   # interlinear annotation separator
    "\ufffb",   # interlinear annotation terminator
})

# Categories of Unicode characters considered "invisible" or confusable.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Mn", "Zs"})


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------

def _check_unicode_tricks(text: str) -> List[Tuple[str, str, int]]:
    """Return a list of ``(category, detail, weight)`` for Unicode anomalies.

    Checks for:
    - Known suspicious zero-width / directional codepoints.
    - High density of invisible-category characters (Category Cf, Mn, Zs
      beyond normal space).
    - Homoglyph-style mixing of multiple Unicode scripts in a single
      word-like token (e.g. Cyrillic ``a`` mixed with Latin).
    """
    hits: List[Tuple[str, str, int]] = []

    # --- Suspicious codepoints ------------------------------------------------
    found_codepoints: list[str] = []
    for ch in text:
        if ch in _SUSPICIOUS_CODEPOINTS:
            found_codepoints.append(f"U+{ord(ch):04X} ({unicodedata.name(ch, 'UNKNOWN')})")
    if found_codepoints:
        unique = sorted(set(found_codepoints))
        count = len(found_codepoints)
        weight = 2 if count < 5 else (3 if count < 15 else 4)
        hits.append((
            "UNICODE_TRICK",
            f"Found {count} suspicious codepoint(s): {', '.join(unique[:5])}"
            + (" ..." if len(unique) > 5 else ""),
            weight,
        ))

    # --- RTL overrides specifically (high-risk) --------------------------------
    rtl_overrides = sum(1 for ch in text if ch in ("\u202e", "\u202d", "\u2066", "\u2067"))
    if rtl_overrides:
        hits.append((
            "UNICODE_TRICK",
            f"Found {rtl_overrides} directional override character(s)",
            4,
        ))

    # --- Invisible character density -------------------------------------------
    invisible_count = 0
    for ch in text:
        if ch not in (" ", "\t", "\n", "\r") and unicodedata.category(ch) in _INVISIBLE_CATEGORIES:
            invisible_count += 1
    text_len = max(len(text), 1)
    ratio = invisible_count / text_len
    if invisible_count > 10 and ratio > 0.02:
        hits.append((
            "UNICODE_TRICK",
            f"High invisible-character density: {invisible_count}/{text_len} "
            f"({ratio:.1%})",
            3,
        ))

    # --- Script mixing (homoglyph detection) -----------------------------------
    # Split into word-like tokens, check if any single token mixes scripts
    # beyond Common/Inherited.
    _IGNORED_SCRIPTS = {"Common", "Inherited"}
    words = re.findall(r"\w{4,}", text)
    mixed_examples: list[str] = []
    for word in words:
        scripts = set()
        for ch in word:
            try:
                script = unicodedata.script(ch)  # type: ignore[attr-defined]
            except AttributeError:
                # unicodedata.script unavailable in older Python versions;
                # fall back to category-based heuristic.
                break
            if script not in _IGNORED_SCRIPTS:
                scripts.add(script)
        else:
            if len(scripts) > 1:
                mixed_examples.append(word)
    if mixed_examples:
        hits.append((
            "UNICODE_TRICK",
            f"Mixed-script tokens (possible homoglyphs): "
            f"{', '.join(repr(w) for w in mixed_examples[:3])}"
            + (" ..." if len(mixed_examples) > 3 else ""),
            3,
        ))

    return hits


def _compute_threat_level(total_score: int, category_count: int) -> str:
    """Map an aggregate weight score and category breadth to a threat level.

    Scoring rationale:
    - A single low-weight hit (score 1-2) from one category is ``"low"``.
    - Multiple matches OR a single high-weight hit is ``"medium"``.
    - Broad multi-category matches or heavy score is ``"high"``.
    - Obvious attacks (very high score or 4+ categories) are ``"critical"``.
    """
    if total_score == 0:
        return "none"
    if total_score <= 2 and category_count == 1:
        return "low"
    if total_score <= 5 and category_count <= 2:
        return "medium"
    if total_score <= 10 and category_count <= 3:
        return "high"
    return "critical"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_for_injection(text: str) -> InjectionResult:
    """Scan *text* for prompt-injection patterns.

    Returns an :class:`InjectionResult` summarising the findings.  The scan
    is intentionally fast (compiled regex + single-pass Unicode check) so it
    can be called on every piece of external content entering the pipeline.
    """
    if not text:
        return InjectionResult()

    categories: list[str] = []
    details: list[str] = []
    total_score = 0

    # Regex patterns
    for pattern, category, description, weight in _PATTERNS:
        if pattern.search(text):
            categories.append(category)
            details.append(description)
            total_score += weight

    # Unicode heuristic checks
    for category, detail, weight in _check_unicode_tricks(text):
        categories.append(category)
        details.append(detail)
        total_score += weight

    # Deduplicate categories while preserving order
    seen: set[str] = set()
    unique_categories: list[str] = []
    for cat in categories:
        if cat not in seen:
            seen.add(cat)
            unique_categories.append(cat)

    threat_level = _compute_threat_level(total_score, len(unique_categories))

    return InjectionResult(
        is_suspicious=total_score > 0,
        threat_level=threat_level,
        categories=unique_categories,
        details=details,
    )


def sanitize_external_content(
    text: str,
    max_length: int = 20_000,
) -> Tuple[str, InjectionResult]:
    """Truncate oversized content and scan for injections.

    Parameters:
        text: Raw external content.
        max_length: Maximum allowed character length.  Content exceeding
            this limit is truncated to ``head + gap_marker + tail`` where
            each piece is roughly ``max_length // 2`` characters.

    Returns:
        A ``(sanitized_text, result)`` tuple.  The sanitized text has been
        length-capped but is **not** stripped of suspicious content (the
        caller decides what to do based on the :class:`InjectionResult`).
    """
    if not text:
        return "", InjectionResult()

    sanitized = text

    if len(sanitized) > max_length:
        half = max_length // 2
        gap_marker = (
            f"\n\n[... truncated {len(sanitized) - max_length:,} characters ...]\n\n"
        )
        sanitized = sanitized[:half] + gap_marker + sanitized[-half:]

    result = scan_for_injection(sanitized)
    return sanitized, result
