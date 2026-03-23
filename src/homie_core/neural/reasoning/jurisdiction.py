"""Jurisdiction-aware context and rule engine.

Auto-detects jurisdiction from user config (timezone / location) and provides
tax rules and legal frameworks for domain reasoning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jurisdiction context
# ---------------------------------------------------------------------------

@dataclass
class JurisdictionContext:
    """Encapsulates the legal / tax jurisdiction for domain reasoning."""

    country: str
    state_province: str = ""
    tax_regime: str = ""
    currency: str = ""
    fiscal_year_start: str = "January"
    legal_framework: str = ""


# ---------------------------------------------------------------------------
# Timezone -> jurisdiction mapping (common cases)
# ---------------------------------------------------------------------------

_TZ_COUNTRY_MAP: dict[str, dict] = {
    "America/New_York": {"country": "US", "state_province": "New York", "tax_regime": "IRS", "currency": "USD", "legal_framework": "US Federal + State"},
    "America/Chicago": {"country": "US", "state_province": "Illinois", "tax_regime": "IRS", "currency": "USD", "legal_framework": "US Federal + State"},
    "America/Los_Angeles": {"country": "US", "state_province": "California", "tax_regime": "IRS", "currency": "USD", "legal_framework": "US Federal + State"},
    "America/Denver": {"country": "US", "state_province": "Colorado", "tax_regime": "IRS", "currency": "USD", "legal_framework": "US Federal + State"},
    "America/Toronto": {"country": "CA", "state_province": "Ontario", "tax_regime": "CRA", "currency": "CAD", "legal_framework": "Canadian Federal + Provincial"},
    "America/Vancouver": {"country": "CA", "state_province": "British Columbia", "tax_regime": "CRA", "currency": "CAD", "legal_framework": "Canadian Federal + Provincial"},
    "Europe/London": {"country": "GB", "state_province": "England", "tax_regime": "HMRC", "currency": "GBP", "legal_framework": "UK Common Law"},
    "Europe/Berlin": {"country": "DE", "state_province": "", "tax_regime": "Finanzamt", "currency": "EUR", "legal_framework": "German Civil Law"},
    "Europe/Paris": {"country": "FR", "state_province": "", "tax_regime": "DGFiP", "currency": "EUR", "legal_framework": "French Civil Law"},
    "Asia/Kolkata": {"country": "IN", "state_province": "", "tax_regime": "GST", "currency": "INR", "fiscal_year_start": "April", "legal_framework": "Indian Common Law"},
    "Asia/Tokyo": {"country": "JP", "state_province": "", "tax_regime": "NTA", "currency": "JPY", "fiscal_year_start": "April", "legal_framework": "Japanese Civil Law"},
    "Asia/Shanghai": {"country": "CN", "state_province": "", "tax_regime": "SAT", "currency": "CNY", "legal_framework": "Chinese Civil Law"},
    "Asia/Singapore": {"country": "SG", "state_province": "", "tax_regime": "IRAS", "currency": "SGD", "legal_framework": "Singapore Common Law"},
    "Australia/Sydney": {"country": "AU", "state_province": "NSW", "tax_regime": "ATO", "currency": "AUD", "fiscal_year_start": "July", "legal_framework": "Australian Common Law"},
    "Australia/Melbourne": {"country": "AU", "state_province": "Victoria", "tax_regime": "ATO", "currency": "AUD", "fiscal_year_start": "July", "legal_framework": "Australian Common Law"},
    "Pacific/Auckland": {"country": "NZ", "state_province": "", "tax_regime": "IRD", "currency": "NZD", "fiscal_year_start": "April", "legal_framework": "New Zealand Common Law"},
}

# Country-code -> broad region prefix for fuzzy timezone matching
_REGION_DEFAULTS: dict[str, dict] = {
    "America": {"country": "US", "tax_regime": "IRS", "currency": "USD", "legal_framework": "US Federal"},
    "Europe": {"country": "EU", "tax_regime": "", "currency": "EUR", "legal_framework": "EU Law"},
    "Asia": {"country": "", "tax_regime": "", "currency": "", "legal_framework": ""},
    "Australia": {"country": "AU", "tax_regime": "ATO", "currency": "AUD", "fiscal_year_start": "July", "legal_framework": "Australian Common Law"},
    "Pacific": {"country": "", "tax_regime": "", "currency": "", "legal_framework": ""},
    "Africa": {"country": "", "tax_regime": "", "currency": "", "legal_framework": ""},
}

# ---------------------------------------------------------------------------
# Tax rules (simplified reference data)
# ---------------------------------------------------------------------------

_TAX_RULES: dict[str, dict] = {
    "IRS": {
        "regime": "IRS",
        "income_tax_brackets": "progressive 10%-37%",
        "capital_gains": "0/15/20% based on income",
        "sales_tax": "state-dependent",
        "filing_deadline": "April 15",
        "fiscal_year_default": "January",
        "common_deductions": ["mortgage interest", "state/local taxes", "charitable", "medical", "business expenses"],
    },
    "HMRC": {
        "regime": "HMRC",
        "income_tax_brackets": "20/40/45%",
        "capital_gains": "10/20%",
        "sales_tax": "VAT 20%",
        "filing_deadline": "January 31",
        "fiscal_year_default": "April",
        "common_deductions": ["pension contributions", "gift aid", "business expenses"],
    },
    "GST": {
        "regime": "GST (India)",
        "income_tax_brackets": "5/20/30%",
        "capital_gains": "STCG 15% / LTCG 10%",
        "sales_tax": "GST 5/12/18/28%",
        "filing_deadline": "July 31",
        "fiscal_year_default": "April",
        "common_deductions": ["80C investments", "HRA", "medical insurance", "home loan interest"],
    },
    "ATO": {
        "regime": "ATO",
        "income_tax_brackets": "19/32.5/37/45%",
        "capital_gains": "50% discount after 12 months",
        "sales_tax": "GST 10%",
        "filing_deadline": "October 31",
        "fiscal_year_default": "July",
        "common_deductions": ["work-related expenses", "self-education", "donations", "investment expenses"],
    },
    "CRA": {
        "regime": "CRA",
        "income_tax_brackets": "15/20.5/26/29/33%",
        "capital_gains": "50% inclusion rate",
        "sales_tax": "GST 5% + provincial",
        "filing_deadline": "April 30",
        "fiscal_year_default": "January",
        "common_deductions": ["RRSP", "childcare", "moving expenses", "union dues"],
    },
}

# ---------------------------------------------------------------------------
# Legal framework descriptions
# ---------------------------------------------------------------------------

_LEGAL_FRAMEWORKS: dict[str, str] = {
    "US Federal + State": "United States federal law with state-level regulations. UCC for commercial transactions, state contract law, federal securities law.",
    "US Federal": "United States federal law. UCC, federal regulations, SEC compliance.",
    "UK Common Law": "English common law. Contract Act, Companies Act, GDPR, Financial Services Act.",
    "Canadian Federal + Provincial": "Canadian federal law with provincial regulations. Common law (except Quebec civil law).",
    "Indian Common Law": "Indian legal system. Contract Act 1872, Companies Act 2013, GST Act, Income Tax Act 1961.",
    "Australian Common Law": "Australian common law. Corporations Act, Competition and Consumer Act, Fair Work Act.",
    "German Civil Law": "German civil law (BGB). Commercial Code (HGB), GmbH Act, EU regulations.",
    "French Civil Law": "French civil law (Code Civil). Commercial Code, EU regulations.",
    "Japanese Civil Law": "Japanese civil law. Commercial Code, Companies Act, Financial Instruments and Exchange Act.",
    "Chinese Civil Law": "Chinese civil law. Contract Law, Company Law, Securities Law.",
    "Singapore Common Law": "Singapore common law based on English law. Companies Act, Securities and Futures Act.",
    "New Zealand Common Law": "New Zealand common law. Contract and Commercial Law Act, Companies Act, Financial Markets Conduct Act.",
    "EU Law": "European Union law framework. GDPR, EU Directives, member state implementation.",
}


class JurisdictionEngine:
    """Detects and provides jurisdiction-specific rules and frameworks."""

    def detect_from_config(self, config: dict | object) -> JurisdictionContext:
        """Auto-detect jurisdiction from config timezone/location.

        *config* can be a dict or any object with ``.get()`` or attribute
        access for ``timezone``, ``location``, ``country``, etc.
        """
        tz = self._extract(config, "timezone", "")
        country = self._extract(config, "country", "")
        state = self._extract(config, "state_province", "")

        # Direct timezone match
        if tz and tz in _TZ_COUNTRY_MAP:
            data = _TZ_COUNTRY_MAP[tz]
            return JurisdictionContext(
                country=country or data.get("country", ""),
                state_province=state or data.get("state_province", ""),
                tax_regime=data.get("tax_regime", ""),
                currency=data.get("currency", ""),
                fiscal_year_start=data.get("fiscal_year_start", "January"),
                legal_framework=data.get("legal_framework", ""),
            )

        # Fuzzy: match by timezone region prefix
        if tz:
            region = tz.split("/")[0] if "/" in tz else ""
            if region in _REGION_DEFAULTS:
                data = _REGION_DEFAULTS[region]
                return JurisdictionContext(
                    country=country or data.get("country", ""),
                    state_province=state,
                    tax_regime=data.get("tax_regime", ""),
                    currency=data.get("currency", ""),
                    fiscal_year_start=data.get("fiscal_year_start", "January"),
                    legal_framework=data.get("legal_framework", ""),
                )

        # Fallback: use whatever country info is available
        return JurisdictionContext(
            country=country or "Unknown",
            state_province=state,
        )

    def get_tax_rules(self, jurisdiction: JurisdictionContext) -> dict:
        """Return tax rule reference data for the given jurisdiction."""
        regime = jurisdiction.tax_regime
        if regime in _TAX_RULES:
            return dict(_TAX_RULES[regime])

        # Try partial match
        for key, rules in _TAX_RULES.items():
            if key.lower() in regime.lower() or regime.lower() in key.lower():
                return dict(rules)

        return {
            "regime": regime or "unknown",
            "note": "No detailed tax rules available for this jurisdiction. Use LLM for analysis.",
        }

    def get_legal_framework(self, jurisdiction: JurisdictionContext) -> str:
        """Return a description of the legal framework for the jurisdiction."""
        fw = jurisdiction.legal_framework
        if fw in _LEGAL_FRAMEWORKS:
            return _LEGAL_FRAMEWORKS[fw]

        # Partial match
        for key, desc in _LEGAL_FRAMEWORKS.items():
            if key.lower() in fw.lower() or fw.lower() in key.lower():
                return desc

        return f"Legal framework: {fw or 'unknown'}. No detailed description available."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(config, key: str, default: str) -> str:
        """Extract a value from config (dict-like or object)."""
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)
