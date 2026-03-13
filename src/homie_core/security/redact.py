"""Secret redaction module for Homie AI.

Detects and masks API keys, tokens, credentials, and other sensitive data
in text before it is displayed to users or written to logs.

All patterns are compiled at module level for performance. The module is
pure-regex with no external dependencies.
"""

from __future__ import annotations

import logging
import re
from typing import List, NamedTuple, Pattern


# ---------------------------------------------------------------------------
# Global toggle — set to False to bypass all redaction (e.g. in debug builds)
# ---------------------------------------------------------------------------
REDACTION_ENABLED: bool = True

# Minimum length above which we preserve a prefix/suffix instead of full mask
_PARTIAL_MASK_THRESHOLD = 18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SecretPattern(NamedTuple):
    """A named regex that captures a secret."""
    name: str
    pattern: Pattern[str]
    # Optional group index that contains the sensitive portion.
    # 0 means the entire match.
    group: int = 0


def _mask(value: str) -> str:
    """Return a masked version of *value*.

    Short tokens (< 18 chars) are replaced entirely with ``***``.
    Longer tokens preserve the first 6 and last 4 characters:
    ``sk-abc...xyz1``.
    """
    if len(value) < _PARTIAL_MASK_THRESHOLD:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


# ---------------------------------------------------------------------------
# Compiled patterns — grouped by provider / category
# ---------------------------------------------------------------------------

_PATTERNS: List[_SecretPattern] = [
    # ---- AI / ML providers ------------------------------------------------
    _SecretPattern(
        "openai_api_key",
        re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    ),
    _SecretPattern(
        "anthropic_api_key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    ),
    _SecretPattern(
        "huggingface_token",
        re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    ),
    _SecretPattern(
        "replicate_token",
        re.compile(r"\br8_[A-Za-z0-9]{20,}"),
    ),

    # ---- GitHub -----------------------------------------------------------
    _SecretPattern(
        "github_pat_classic",
        re.compile(r"\b(?:ghp_|gho_|ghs_)[A-Za-z0-9]{36,}"),
    ),
    _SecretPattern(
        "github_pat_fine",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    ),

    # ---- Slack ------------------------------------------------------------
    _SecretPattern(
        "slack_token",
        re.compile(r"\bxox[bpoas]-[A-Za-z0-9\-]{20,}"),
    ),

    # ---- Google -----------------------------------------------------------
    _SecretPattern(
        "google_api_key",
        re.compile(r"\bAIza[A-Za-z0-9_\-]{30,}"),
    ),

    # ---- Stripe -----------------------------------------------------------
    _SecretPattern(
        "stripe_secret_key",
        re.compile(r"\b(?:sk_live_|sk_test_|rk_live_|rk_test_)[A-Za-z0-9]{20,}"),
    ),

    # ---- AWS --------------------------------------------------------------
    _SecretPattern(
        "aws_access_key_id",
        re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    ),

    # ---- npm / PyPI -------------------------------------------------------
    _SecretPattern(
        "npm_token",
        re.compile(r"\bnpm_[A-Za-z0-9]{20,}"),
    ),
    _SecretPattern(
        "pypi_token",
        re.compile(r"\bpypi-[A-Za-z0-9_\-]{20,}"),
    ),

    # ---- DigitalOcean -----------------------------------------------------
    _SecretPattern(
        "digitalocean_pat",
        re.compile(r"\bdop_v1_[a-f0-9]{64}"),
    ),

    # ---- Telegram ---------------------------------------------------------
    _SecretPattern(
        "telegram_bot_token",
        re.compile(r"\bbot\d{8,}:[A-Za-z0-9_\-]{30,}"),
    ),

    # ---- Bearer / Authorization header ------------------------------------
    _SecretPattern(
        "bearer_token",
        re.compile(r"(?i)\bBearer\s+([A-Za-z0-9_\-\.]{20,})"),
        group=1,
    ),

    # ---- Private key blocks -----------------------------------------------
    _SecretPattern(
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END (?:RSA |EC )?PRIVATE KEY-----"
        ),
    ),

    # ---- Database connection strings with embedded passwords --------------
    _SecretPattern(
        "database_uri",
        re.compile(
            r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)"
            r"://[^:]+:([^@\s]{1,})@[^\s]+"
        ),
        group=1,
    ),

    # ---- Environment variable assignments with secret-like names ----------
    _SecretPattern(
        "env_secret_assignment",
        re.compile(
            r"(?i)(?:API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH)"
            r"[\w]*\s*[=:]\s*['\"]?([A-Za-z0-9_\-\.\/\+]{8,})['\"]?"
        ),
        group=1,
    ),

    # ---- JSON fields with secret-like keys --------------------------------
    _SecretPattern(
        "json_secret_field",
        re.compile(
            r'(?i)"(?:api_?key|token|password|access_token|secret'
            r'|client_secret|refresh_token|auth_token)"'
            r'\s*:\s*"([^"]{8,})"'
        ),
        group=1,
    ),

    # ---- E.164 phone numbers ----------------------------------------------
    _SecretPattern(
        "phone_e164",
        re.compile(r"(?<![A-Za-z0-9])\+[1-9]\d{6,14}(?![A-Za-z0-9])"),
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def redact_sensitive_text(text: str) -> str:
    """Return *text* with detected secrets replaced by masked equivalents.

    When ``REDACTION_ENABLED`` is ``False`` the input is returned unchanged.

    Examples::

        >>> redact_sensitive_text("key is sk-abc123def456ghi789jkl0123456789XY")
        'key is sk-abc...9XY'

        >>> redact_sensitive_text("ghp_Short1234567")
        '***'
    """
    if not REDACTION_ENABLED:
        return text

    result = text

    for sp in _PATTERNS:
        def _replace(match: re.Match, *, _sp: _SecretPattern = sp) -> str:
            secret = match.group(_sp.group)
            masked = _mask(secret)
            if _sp.group == 0:
                return masked
            # Replace only the captured group within the full match
            full = match.group(0)
            return full.replace(secret, masked, 1)

        result = sp.pattern.sub(_replace, result)

    return result


class RedactingFormatter(logging.Formatter):
    """A :class:`logging.Formatter` that redacts secrets from log records.

    Drop-in replacement for the standard formatter::

        handler = logging.StreamHandler()
        handler.setFormatter(RedactingFormatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)

    The formatter delegates to :func:`redact_sensitive_text` so all
    configured secret patterns are applied transparently.
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: str = "%",
        validate: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record, then redact any secrets found."""
        formatted = super().format(record)
        return redact_sensitive_text(formatted)
