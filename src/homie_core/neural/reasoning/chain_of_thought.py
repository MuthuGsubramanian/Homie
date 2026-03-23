"""Chain-of-thought structured reasoning pipeline.

Provides step-by-step reasoning for complex questions and domain-specific
analysis with jurisdiction awareness.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from homie_core.neural.reasoning.jurisdiction import JurisdictionContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_REASON_PROMPT = """\
Think through the following question step by step.

Context:
{context}

Question:
{question}

Return a JSON object with:
- "steps": list of objects, each with "step_number" (int), "reasoning" (str), "conclusion" (str)
- "final_answer": string with the overall answer
- "confidence": float 0-1 indicating your confidence
- "assumptions": list of strings noting any assumptions made

Only return the JSON object, no other text.
"""

_DOMAIN_ANALYSIS_PROMPT = """\
You are a {domain} expert operating under the following jurisdiction:
- Country: {country}
- State/Province: {state_province}
- Tax regime: {tax_regime}
- Currency: {currency}
- Fiscal year starts: {fiscal_year_start}
- Legal framework: {legal_framework}

Analyze the following content using chain-of-thought reasoning.

Content:
{content}

Return a JSON object with:
- "domain": the domain name
- "jurisdiction": object with country and relevant details
- "reasoning_steps": list of objects each with "step" (int), "analysis" (str), "finding" (str)
- "summary": overall summary string
- "key_insights": list of strings
- "risk_assessment": object with "level" (low/medium/high), "factors" (list of strings)
- "recommendations": list of strings
- "confidence": float 0-1

Only return the JSON object, no other text.
"""


class ChainOfThought:
    """Structured reasoning engine using chain-of-thought prompting."""

    def __init__(self, inference_fn: Callable[[str], str]) -> None:
        self._infer = inference_fn

    def reason(self, question: str, context: dict | None = None) -> dict:
        """Reason through *question* step-by-step given optional *context*.

        Returns a dict with ``steps``, ``final_answer``, ``confidence``,
        and ``assumptions``.
        """
        if not question or not question.strip():
            return {
                "steps": [],
                "final_answer": "",
                "confidence": 0.0,
                "assumptions": [],
            }

        ctx_str = json.dumps(context, indent=2) if context else "No additional context."
        prompt = _REASON_PROMPT.format(context=ctx_str, question=question)
        raw = self._infer(prompt).strip()

        return self._parse_json(raw, fallback={
            "steps": [{"step_number": 1, "reasoning": raw, "conclusion": raw}],
            "final_answer": raw,
            "confidence": 0.5,
            "assumptions": [],
        })

    def analyze_with_domain(
        self,
        content: str,
        domain: str,
        jurisdiction: JurisdictionContext | None = None,
    ) -> dict:
        """Analyze *content* with domain expertise and jurisdiction context.

        Returns a structured analysis with reasoning steps, risk assessment,
        and recommendations.
        """
        if not content or not content.strip():
            return {
                "domain": domain,
                "jurisdiction": {},
                "reasoning_steps": [],
                "summary": "",
                "key_insights": [],
                "risk_assessment": {"level": "low", "factors": []},
                "recommendations": [],
                "confidence": 0.0,
            }

        if jurisdiction is None:
            jurisdiction = JurisdictionContext(country="Unknown")

        prompt = _DOMAIN_ANALYSIS_PROMPT.format(
            domain=domain,
            country=jurisdiction.country,
            state_province=jurisdiction.state_province,
            tax_regime=jurisdiction.tax_regime,
            currency=jurisdiction.currency,
            fiscal_year_start=jurisdiction.fiscal_year_start,
            legal_framework=jurisdiction.legal_framework,
            content=content[:6000],
        )
        raw = self._infer(prompt).strip()

        return self._parse_json(raw, fallback={
            "domain": domain,
            "jurisdiction": {"country": jurisdiction.country},
            "reasoning_steps": [{"step": 1, "analysis": raw, "finding": raw}],
            "summary": raw,
            "key_insights": [],
            "risk_assessment": {"level": "medium", "factors": []},
            "recommendations": [],
            "confidence": 0.5,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str, fallback: dict) -> dict:
        """Best-effort parse of a JSON object from LLM output."""
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start:end + 1])
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse chain-of-thought JSON from LLM output")
            return fallback
