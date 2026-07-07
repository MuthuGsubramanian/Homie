"""Fuzzy Functions plugin for HOMIE (Program-as-Weights, arXiv:2607.02512v1).

Everyday tasks such as flagging important log lines, repairing malformed
JSON, or ranking candidates by intent resist crisp rule-based code and are
often outsourced to cloud LLM APIs at the cost of locality, reproducibility,
and price.  This plugin brings the "Program-as-Weights" paradigm to HOMIE:
each task is a *fuzzy function* -- a small, locally-executable callable
compiled from a natural-language specification.

Execution modes per function:

1. ``heuristic`` (default) -- a compact, dependency-free scoring routine
   distilled from the specification.  Deterministic, offline, and fast.
2. ``llm`` (opt-in via ``fuzzy.use_llm: true`` in ``homie.config.yaml``) --
   low-confidence heuristic results escalate to the locally configured
   Ollama model (``homie.llm_ollama``); answers are memoized on disk under
   ``~/.homie/fuzzy_cache`` so repeated calls are reproducible and free.

No remote network calls are ever made: the LLM backend only talks to the
local Ollama endpoint, and only when explicitly enabled in config.

Example:
    import json
    from homie.plugins.fuzzy_functions import register

    plugin = register(cfg)  # cfg optional; heuristics work without it

    hit = plugin.call("log_alert", "kernel: Out of memory: Kill process 4242")
    if hit.value:
        ...  # alert the operator

    fixed = plugin.call("json_repair", "{'name': 'msi', 'up': True,}")
    ranked = plugin.call(
        "rank_by_intent",
        json.dumps({"intent": "free disk space",
                    "candidates": ["df -h", "uptime", "apt-get clean"]}),
    )
    plugin.deactivate()
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from homie.config import HomieConfig, cfg_get

DEFAULT_CACHE_DIR = Path.home() / ".homie" / "fuzzy_cache"
DEFAULT_ESCALATE_BELOW = 0.5


@dataclass
class FuzzyResult:
    """Outcome of one fuzzy-function invocation."""

    name: str
    value: Any
    confidence: float
    source: str  # "heuristic" | "llm" | "cache"


def _default_parse(raw: str) -> str:
    return raw.strip()


@dataclass
class FuzzyFunction:
    """A locally-executable fuzzy function compiled from a natural-language spec.

    ``heuristic`` is the offline approximation of the spec (the "weights"):
    it maps input text to ``(value, confidence)``.  ``parse_output`` converts
    raw LLM text into the function's return type when the opt-in LLM backend
    handles an escalated call.
    """

    name: str
    spec: str
    heuristic: Callable[[str], Tuple[Any, float]]
    parse_output: Callable[[str], Any] = field(default=_default_parse)

    def run_heuristic(self, text: str) -> FuzzyResult:
        value, confidence = self.heuristic(text)
        return FuzzyResult(self.name, value, confidence, "heuristic")

    def build_prompt(self, text: str) -> str:
        return (
            "You are a compact fuzzy function. Follow the specification "
            "exactly and return ONLY the answer, no prose, no markdown.\n\n"
            f"Specification: {self.spec}\n\nInput:\n{text}\n"
        )


def _cache_key(fn: FuzzyFunction, text: str) -> str:
    digest = hashlib.sha256(f"{fn.name}|{fn.spec}|{text}".encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Built-in heuristics (the "weights" distilled from each specification)
# ---------------------------------------------------------------------------

_ALERT_PATTERNS: List[Tuple[Any, float]] = [
    (re.compile(r"\b(panic|fatal|critical|emerg)\b", re.I), 0.9),
    (re.compile(r"out of memory|oom[- ]?kill|segfault|core dumped", re.I), 0.9),
    (re.compile(r"disk full|no space left|read[- ]?only file ?system", re.I), 0.9),
    (re.compile(r"\b(error|exception|traceback|unauthorized|denied)\b", re.I), 0.6),
    (re.compile(r"\b(fail(ed|ure)?|refused|unreachable|timeout|timed out)\b", re.I), 0.55),
    (re.compile(r"\b(warn(ing)?|deprecated|retry(ing)?)\b", re.I), 0.3),
]


def _log_alert_heuristic(text: str) -> Tuple[bool, float]:
    """Score one log line; alert when the strongest matching pattern >= 0.5.

    Confidence is distance from the decision boundary, so clearly benign
    (score 0.0) and clearly critical (score 0.9) lines are both confident,
    while borderline lines escalate to the LLM backend when enabled.
    """
    score = 0.0
    for pattern, weight in _ALERT_PATTERNS:
        if pattern.search(text):
            score = max(score, weight)
    important = score >= 0.5
    confidence = round(min(1.0, abs(score - 0.5) * 2.0), 2)
    return important, confidence


_FENCE_RE = re.compile(r"```(?:json)?", re.I)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)")


def _extract_blob(text: str) -> str:
    """Isolate the outermost JSON object/array substring (mirrors ai_planner)."""
    cleaned = _FENCE_RE.sub("", text).strip().strip("`").strip()
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        return cleaned
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end > start:
        return cleaned[start : end + 1]
    return cleaned


def _json_repair_heuristic(text: str) -> Tuple[Optional[Any], float]:
    """Try progressively more aggressive repairs; confidence drops per step."""
    candidate = _extract_blob(text)
    fixed = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    fixed = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    attempts = [(candidate, 1.0), (fixed, 0.8), (fixed.replace("'", '"'), 0.6)]
    for attempt, confidence in attempts:
        try:
            return json.loads(attempt), confidence
        except (json.JSONDecodeError, TypeError):
            continue
    return None, 0.0


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _rank_by_intent_heuristic(text: str) -> Tuple[Optional[List[str]], float]:
    """Rank candidates by Jaccard token overlap with the intent."""
    try:
        payload = json.loads(text)
        intent = str(payload["intent"])
        candidates = [str(c) for c in payload["candidates"]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, 0.0
    intent_tokens = _tokens(intent)

    def overlap(candidate: str) -> float:
        cand_tokens = _tokens(candidate)
        union = intent_tokens | cand_tokens
        if not union:
            return 0.0
        return len(intent_tokens & cand_tokens) / len(union)

    ranked = sorted(candidates, key=overlap, reverse=True)
    top = overlap(ranked[0]) if ranked else 0.0
    return ranked, round(min(1.0, 0.3 + top), 2)


def _parse_yes_no(raw: str) -> bool:
    return "yes" in raw.lower()[:16]


def _parse_json_value(raw: str) -> Any:
    value, _confidence = _json_repair_heuristic(raw)
    return value


def _builtin_functions() -> List[FuzzyFunction]:
    """Fuzzy functions shipped with the plugin, one per paper use-case."""
    return [
        FuzzyFunction(
            name="log_alert",
            spec=(
                "Given a single log line, answer 'yes' if a human operator "
                "should be alerted (crashes, errors, resource exhaustion, "
                "security denials) and 'no' otherwise."
            ),
            heuristic=_log_alert_heuristic,
            parse_output=_parse_yes_no,
        ),
        FuzzyFunction(
            name="json_repair",
            spec=(
                "Given malformed JSON, return only the corrected JSON. Fix "
                "quoting, trailing commas, Python literals, and stray text."
            ),
            heuristic=_json_repair_heuristic,
            parse_output=_parse_json_value,
        ),
        FuzzyFunction(
            name="rank_by_intent",
            spec=(
                'Given JSON {"intent": str, "candidates": [str]}, return the '
                "candidates as a JSON list sorted from most to least relevant "
                "to the intent."
            ),
            heuristic=_rank_by_intent_heuristic,
            parse_output=_parse_json_value,
        ),
    ]


class FuzzyFunctionsPlugin:
    """Registers fuzzy functions and dispatches calls with optional LLM escalation."""

    def __init__(
        self,
        cfg: Optional[HomieConfig] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._functions: Dict[str, FuzzyFunction] = {}
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Register the built-in fuzzy functions and enable dispatch."""
        for fn in _builtin_functions():
            self._functions[fn.name] = fn
        self._active = True
        logging.info(
            "fuzzy_functions plugin active (%d functions, llm=%s)",
            len(self._functions),
            self._llm_enabled(),
        )

    def deactivate(self) -> None:
        """Unregister all functions and disable dispatch."""
        self._functions.clear()
        self._active = False
        logging.info("fuzzy_functions plugin deactivated")

    def register_function(self, fn: FuzzyFunction) -> None:
        """Add a custom fuzzy function (replaces any same-named function)."""
        self._functions[fn.name] = fn

    def names(self) -> List[str]:
        return sorted(self._functions)

    def call(self, name: str, text: str) -> FuzzyResult:
        """Run fuzzy function ``name`` on ``text``.

        The heuristic runs first; a confident result is returned directly.
        Low-confidence results escalate to the local Ollama model only when
        ``fuzzy.use_llm`` is true, with on-disk memoization for repeatability.
        """
        if not self._active:
            raise RuntimeError("fuzzy_functions plugin is not active; call activate() first")
        fn = self._functions.get(name)
        if fn is None:
            raise KeyError(f"Unknown fuzzy function '{name}'. Available: {self.names()}")
        result = fn.run_heuristic(text)
        if result.confidence >= self._escalate_below() or not self._llm_enabled():
            return result
        return self._call_llm(fn, text, fallback=result)

    def _llm_enabled(self) -> bool:
        return bool(self.cfg and cfg_get(self.cfg, "fuzzy", "use_llm", default=False))

    def _escalate_below(self) -> float:
        if not self.cfg:
            return DEFAULT_ESCALATE_BELOW
        return float(
            cfg_get(self.cfg, "fuzzy", "escalate_below", default=DEFAULT_ESCALATE_BELOW)
        )

    def _call_llm(self, fn: FuzzyFunction, text: str, fallback: FuzzyResult) -> FuzzyResult:
        cached = self._cache_read(fn, text)
        if cached is not None:
            return FuzzyResult(fn.name, cached, 0.9, "cache")
        try:
            from homie.llm_ollama import ollama_generate

            raw = ollama_generate(self.cfg, fn.build_prompt(text))
            value = fn.parse_output(raw)
        except Exception as exc:  # noqa: BLE001 -- LLMError, parse errors alike
            logging.warning(
                "fuzzy '%s': LLM escalation failed, keeping heuristic: %s", fn.name, exc
            )
            return fallback
        self._cache_write(fn, text, value)
        return FuzzyResult(fn.name, value, 0.9, "llm")

    def _cache_path(self, fn: FuzzyFunction, text: str) -> Path:
        return self.cache_dir / f"{_cache_key(fn, text)}.json"

    def _cache_read(self, fn: FuzzyFunction, text: str) -> Optional[Any]:
        path = self._cache_path(fn, text)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["value"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _cache_write(self, fn: FuzzyFunction, text: str, value: Any) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(fn, text).write_text(
                json.dumps({"name": fn.name, "value": value}, ensure_ascii=True),
                encoding="utf-8",
            )
        except (TypeError, OSError):
            logging.debug("fuzzy cache write skipped for '%s'", fn.name)


_PLUGIN: Optional[FuzzyFunctionsPlugin] = None


def register(cfg: Optional[HomieConfig] = None) -> FuzzyFunctionsPlugin:
    """Create (or reuse) the module-level plugin instance and activate it."""
    global _PLUGIN
    if _PLUGIN is None:
        _PLUGIN = FuzzyFunctionsPlugin(cfg)
    if not _PLUGIN.active:
        _PLUGIN.activate()
    return _PLUGIN


__all__ = [
    "FuzzyResult",
    "FuzzyFunction",
    "FuzzyFunctionsPlugin",
    "register",
    "DEFAULT_CACHE_DIR",
]
