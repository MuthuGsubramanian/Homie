"""Financial data models for analysis results."""
from __future__ import annotations

from dataclasses import dataclass, field


def parse_amount(amount: str | None) -> float | None:
    """Safely parse amount string to float."""
    if not amount:
        return None
    try:
        return float(amount.replace(",", ""))
    except (ValueError, TypeError):
        return None


@dataclass
class SpendingSummary:
    period: str
    total_amount: float
    currency: str
    by_category: dict[str, float] = field(default_factory=dict)
    record_count: int = 0


@dataclass
class BillGroup:
    """A group of recurring bills from the same source."""
    description_pattern: str
    typical_amount: float
    currency: str
    occurrences: int
    last_due: float | None = None
    record_ids: list[int] = field(default_factory=list)


@dataclass
class TrendResult:
    direction: str  # "up", "down", "stable"
    current_month: float
    previous_month: float
    change_pct: float
    currency: str
