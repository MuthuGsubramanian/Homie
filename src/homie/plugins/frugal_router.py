"""Frugal model router plugin for HOMIE.

Inspired by Frugon (https://github.com/Rodiun/frugon, MIT licence): find
which LLM calls a cheaper model could have handled. The plugin observes
every generation request that flows through
:func:`homie.llm_ollama.ollama_generate`, scores prompt complexity with
fast local heuristics, and appends a verdict to a JSONL ledger under
``~/.homie``. Over time the ledger shows what share of HOMIE's traffic a
smaller local model could serve; when ``auto_downshift`` is enabled the
plugin also reroutes low-complexity prompts to that model transparently.

Runs 100% locally. The plugin itself never opens a network connection;
the only network activity remains HOMIE's own call to the local Ollama
server, and rerouting only happens after explicit opt-in via config.

Configuration (all optional, under ``llm.frugal`` in homie.config.yaml)::

    llm:
      frugal:
        enabled: true
        cheap_model: "qwen2.5:3b"   # small local model to downshift to
        auto_downshift: false       # advise-only unless set to true
        threshold: 0.35             # score below this => cheap tier ok
        ledger_path: "~/.homie/frugal_ledger.jsonl"

Usage::

    from homie.plugins.frugal_router import activate, deactivate, report

    activate(cfg)      # start observing (and rerouting, if opted in)
    ...                # normal HOMIE usage
    print(report())    # human-readable savings summary
    deactivate()       # restore the original generate function
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from homie.config import HomieConfig, cfg_get

DEFAULT_LEDGER = Path.home() / ".homie" / "frugal_ledger.jsonl"
DEFAULT_THRESHOLD = 0.35

# Modules that hold a reference to ollama_generate. ai_planner imports the
# function by name, so its module-level binding must be patched as well.
_PATCH_SITES = ("homie.llm_ollama", "homie.ai_planner")

# Signals that a prompt genuinely needs the stronger model.
_HARD_PATTERNS = (
    r"\bstep[- ]by[- ]step\b",
    r"\bprove\b|\btheorem\b",
    r"\brefactor\b|\bdebug\b",
    r"\bmulti[- ]file\b",
    r"```",
    r"\bwrite (a |an )?(function|class|script|module)\b",
)

# Signals of mechanical, low-entropy work a small model handles well.
_EASY_PATTERNS = (
    r"return only valid json",
    r"fix this to valid json",
    r"\bsummar(y|ise|ize)\b",
    r"\bclassif(y|ication)\b|\blabel\b",
    r"\bextract\b",
    r"\byes or no\b",
    r"\btranslate\b",
)


@dataclass
class CallRecord:
    """One observed LLM call and the frugality verdict for it."""

    ts: float
    model: str
    prompt_chars: int
    score: float
    verdict: str  # "cheap_ok" or "needs_big"
    downshifted: bool


def score_prompt(prompt: str) -> float:
    """Return a complexity score in [0, 1]; higher means it needs the big model.

    Purely heuristic and instant: length pressure plus keyword signals.
    Deliberately biased toward "needs_big" so downshifting stays safe.
    """
    text = prompt.lower()
    score = 0.15 + min(len(prompt) / 8000.0, 0.35)
    for pattern in _HARD_PATTERNS:
        if re.search(pattern, text):
            score += 0.15
    for pattern in _EASY_PATTERNS:
        if re.search(pattern, text):
            score -= 0.12
    return max(0.0, min(1.0, score))


class FrugalRouter:
    """Scores prompts, keeps a local ledger, optionally downshifts models."""

    def __init__(self, cfg: HomieConfig):
        self.cfg = cfg
        self.enabled = bool(cfg_get(cfg, "llm", "frugal", "enabled", default=True))
        self.cheap_model: Optional[str] = cfg_get(
            cfg, "llm", "frugal", "cheap_model", default=None
        )
        self.auto_downshift = bool(
            cfg_get(cfg, "llm", "frugal", "auto_downshift", default=False)
        )
        self.threshold = float(
            cfg_get(cfg, "llm", "frugal", "threshold", default=DEFAULT_THRESHOLD)
        )
        ledger = cfg_get(cfg, "llm", "frugal", "ledger_path", default=None)
        self.ledger_path = Path(ledger).expanduser() if ledger else DEFAULT_LEDGER
        self.records: List[CallRecord] = []

    def route(self, cfg: HomieConfig, prompt: str) -> Tuple[HomieConfig, CallRecord]:
        """Decide which config (and therefore model) to use for this prompt."""
        default_model = cfg_get(cfg, "llm", "model", default="glm-4.7-flash")
        score = score_prompt(prompt)
        cheap_ok = score < self.threshold
        downshift = (
            cheap_ok
            and self.auto_downshift
            and bool(self.cheap_model)
            and self.cheap_model != default_model
        )
        record = CallRecord(
            ts=time.time(),
            model=self.cheap_model if downshift else default_model,
            prompt_chars=len(prompt),
            score=round(score, 3),
            verdict="cheap_ok" if cheap_ok else "needs_big",
            downshifted=downshift,
        )
        self._persist(record)
        if not downshift:
            return cfg, record
        raw = dict(cfg.raw)
        raw["llm"] = dict(raw.get("llm", {}), model=self.cheap_model)
        return HomieConfig(raw=raw, path=cfg.path), record

    def _persist(self, record: CallRecord) -> None:
        self.records.append(record)
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record)) + "\n")
        except OSError:
            # Ledger is best-effort observability; never block a live call.
            logging.debug("frugal_router: could not write ledger", exc_info=True)

    def summary(self) -> Dict[str, object]:
        total = len(self.records)
        cheap = sum(1 for r in self.records if r.verdict == "cheap_ok")
        downshifted = sum(1 for r in self.records if r.downshifted)
        return {
            "calls_observed": total,
            "cheap_model_sufficient": cheap,
            "cheap_share_pct": round(100.0 * cheap / total, 1) if total else 0.0,
            "downshifted": downshifted,
            "auto_downshift": self.auto_downshift,
            "cheap_model": self.cheap_model,
            "ledger_path": str(self.ledger_path),
        }


_router: Optional[FrugalRouter] = None
_original: Optional[Callable[[HomieConfig, str], str]] = None


def _wrapped_generate(cfg: HomieConfig, prompt: str) -> str:
    """Drop-in replacement for ollama_generate with frugal routing."""
    if _router is None or _original is None or not _router.enabled:
        # Defensive: behave exactly like the original if state is missing.
        from homie.llm_ollama import ollama_generate as fallback

        return (_original or fallback)(cfg, prompt)
    routed_cfg, record = _router.route(cfg, prompt)
    if record.downshifted:
        logging.info(
            "frugal_router: downshifting to %s (score=%.2f)", record.model, record.score
        )
    return _original(routed_cfg, prompt)


def activate(cfg: HomieConfig) -> FrugalRouter:
    """Install the frugal router around HOMIE's Ollama generate function."""
    global _router, _original
    if _router is not None:
        return _router  # already active
    from homie.llm_ollama import ollama_generate as original

    _router = FrugalRouter(cfg)
    _original = original
    for name in _PATCH_SITES:
        module = sys.modules.get(name)
        if module is not None and getattr(module, "ollama_generate", None) is original:
            module.ollama_generate = _wrapped_generate
    logging.info(
        "frugal_router active | cheap_model=%s auto_downshift=%s threshold=%.2f",
        _router.cheap_model,
        _router.auto_downshift,
        _router.threshold,
    )
    return _router


def deactivate() -> None:
    """Restore the original generate function and stop recording."""
    global _router, _original
    if _original is None:
        return
    for name in _PATCH_SITES:
        module = sys.modules.get(name)
        if module is not None and getattr(module, "ollama_generate", None) is _wrapped_generate:
            module.ollama_generate = _original
    logging.info("frugal_router deactivated")
    _router = None
    _original = None


def report() -> str:
    """Human-readable savings report for the current session."""
    if _router is None:
        return "frugal_router is not active"
    s = _router.summary()
    return (
        f"Frugal router: {s['calls_observed']} calls observed, "
        f"{s['cheap_model_sufficient']} ({s['cheap_share_pct']}%) could use a cheaper model; "
        f"{s['downshifted']} downshifted (auto_downshift={s['auto_downshift']}). "
        f"Ledger: {s['ledger_path']}"
    )


__all__ = [
    "CallRecord",
    "FrugalRouter",
    "score_prompt",
    "activate",
    "deactivate",
    "report",
]
