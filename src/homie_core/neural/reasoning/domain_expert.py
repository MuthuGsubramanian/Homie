"""Pluggable domain expert module for the ReasoningAgent.

Uses LLM inference to classify domains, extract entities, and apply
domain-specific rules (accounting, finance, legal, tax, general).
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from homie_core.neural.reasoning.jurisdiction import JurisdictionContext

logger = logging.getLogger(__name__)

VALID_DOMAINS = ("accounting", "finance", "legal", "tax", "general")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """\
Classify the following text into exactly one domain.

Domains: accounting, finance, legal, tax, general

Rules:
- accounting: bookkeeping, ledgers, reconciliation, journal entries, chart of accounts
- finance: budgets, cash-flow, investments, financial ratios, forecasting
- legal: contracts, clauses, compliance, litigation, regulations
- tax: tax returns, deductions, filing, withholding, tax codes
- general: anything that does not clearly fit the above

Respond with ONLY the domain name (one word, lowercase).

Text:
{text}
"""

_EXTRACT_PROMPT = """\
Extract structured entities from the following {domain} document.

Return a JSON array of objects. Each object MUST have:
- "type": one of "amount", "date", "party", "clause", "line_item", "reference", "percentage"
- "value": the extracted value as a string
- "context": a short phrase describing where/why this entity matters

Only return the JSON array, no other text.

Document:
{content}
"""

_ANALYZE_PROMPT = """\
You are a {domain} domain expert. Analyze the following document content and provide
a structured analysis.

Return a JSON object with:
- "summary": 2-3 sentence overview
- "key_findings": list of strings
- "risk_flags": list of strings (potential issues or concerns)
- "action_items": list of strings (recommended next steps)
- "confidence": float 0-1 indicating analysis confidence

Only return the JSON object, no other text.

Document:
{content}
"""

_RULES_PROMPT = """\
You are a {domain} expert analyzing extracted entities under the following jurisdiction:
- Country: {country}
- State/Province: {state_province}
- Tax regime: {tax_regime}
- Currency: {currency}
- Fiscal year starts: {fiscal_year_start}
- Legal framework: {legal_framework}

Entities:
{entities}

Apply relevant {domain} rules for this jurisdiction and return a JSON object with:
- "applicable_rules": list of rule descriptions that apply
- "compliance_status": "compliant", "non_compliant", or "needs_review"
- "findings": list of specific findings from applying rules
- "recommendations": list of action items
- "risk_level": "low", "medium", or "high"

Only return the JSON object, no other text.
"""


class DomainExpert:
    """Pluggable domain expert that uses LLM inference for classification,
    entity extraction, and domain-specific analysis."""

    def __init__(self, inference_fn: Callable[[str], str]) -> None:
        self._infer = inference_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_domain(self, text: str) -> str:
        """Classify *text* into one of the supported domains.

        Returns one of: ``accounting``, ``finance``, ``legal``, ``tax``,
        ``general``.
        """
        if not text or not text.strip():
            return "general"

        prompt = _CLASSIFY_PROMPT.format(text=text[:4000])
        raw = self._infer(prompt).strip().lower()

        # The LLM should return a single word; guard against verbosity.
        for domain in VALID_DOMAINS:
            if domain in raw:
                return domain
        return "general"

    def extract_entities(self, text: str, domain: str) -> list[dict]:
        """Extract structured entities (amounts, dates, parties, clauses, etc.)
        from *text* within the given *domain*."""
        if not text or not text.strip():
            return []

        domain = domain if domain in VALID_DOMAINS else "general"
        prompt = _EXTRACT_PROMPT.format(domain=domain, content=text[:6000])
        raw = self._infer(prompt).strip()

        return self._parse_json_array(raw)

    def analyze_document(self, content: str, domain: str) -> dict:
        """Produce a structured analysis of *content* for *domain*."""
        if not content or not content.strip():
            return {
                "summary": "",
                "key_findings": [],
                "risk_flags": [],
                "action_items": [],
                "confidence": 0.0,
            }

        domain = domain if domain in VALID_DOMAINS else "general"
        prompt = _ANALYZE_PROMPT.format(domain=domain, content=content[:6000])
        raw = self._infer(prompt).strip()

        return self._parse_json_object(raw, fallback={
            "summary": raw,
            "key_findings": [],
            "risk_flags": [],
            "action_items": [],
            "confidence": 0.5,
        })

    def apply_rules(
        self,
        entities: list[dict],
        domain: str,
        jurisdiction: dict | JurisdictionContext | None = None,
    ) -> dict:
        """Apply domain-specific rules to *entities* under *jurisdiction*.

        *jurisdiction* can be a :class:`JurisdictionContext` instance or a
        plain ``dict`` with the same keys.
        """
        if not entities:
            return {
                "applicable_rules": [],
                "compliance_status": "needs_review",
                "findings": [],
                "recommendations": [],
                "risk_level": "low",
            }

        domain = domain if domain in VALID_DOMAINS else "general"

        # Normalise jurisdiction to a dict.
        if jurisdiction is None:
            jur = {}
        elif hasattr(jurisdiction, "__dataclass_fields__"):
            from dataclasses import asdict
            jur = asdict(jurisdiction)
        else:
            jur = dict(jurisdiction)

        prompt = _RULES_PROMPT.format(
            domain=domain,
            country=jur.get("country", "Unknown"),
            state_province=jur.get("state_province", ""),
            tax_regime=jur.get("tax_regime", ""),
            currency=jur.get("currency", ""),
            fiscal_year_start=jur.get("fiscal_year_start", "January"),
            legal_framework=jur.get("legal_framework", ""),
            entities=json.dumps(entities, indent=2),
        )
        raw = self._infer(prompt).strip()

        return self._parse_json_object(raw, fallback={
            "applicable_rules": [],
            "compliance_status": "needs_review",
            "findings": [raw],
            "recommendations": [],
            "risk_level": "medium",
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_array(raw: str) -> list[dict]:
        """Best-effort parse of a JSON array from LLM output."""
        try:
            # Try to find the array within the response
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                return json.loads(raw[start:end + 1])
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse entity JSON from LLM output")
            return []

    @staticmethod
    def _parse_json_object(raw: str, fallback: dict) -> dict:
        """Best-effort parse of a JSON object from LLM output."""
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start:end + 1])
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse JSON object from LLM output")
            return fallback
