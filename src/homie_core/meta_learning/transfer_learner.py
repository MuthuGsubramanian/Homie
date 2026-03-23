# src/homie_core/meta_learning/transfer_learner.py
"""Transfer Learner — applies lessons learned in one domain to another."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class _DomainSolution:
    """A recorded solution with its originating domain."""

    domain: str
    task_description: str
    solution: dict
    tags: list[str] = field(default_factory=list)
    success_score: float = 0.0


class TransferLearner:
    """Applies lessons from one domain to another by finding analogies."""

    def __init__(self, knowledge_query: Callable[..., Any] | None = None):
        self._knowledge_query = knowledge_query
        self._solutions: list[_DomainSolution] = []

    # ── public API ──────────────────────────────────────────────────────

    def record_solution(
        self,
        domain: str,
        task_description: str,
        solution: dict,
        tags: list[str] | None = None,
        success_score: float = 1.0,
    ) -> None:
        """Store a successful solution for future transfer."""
        self._solutions.append(
            _DomainSolution(
                domain=domain,
                task_description=task_description,
                solution=solution,
                tags=tags or [],
                success_score=success_score,
            )
        )

    def find_analogies(
        self, task: str, source_domain: str, target_domain: str
    ) -> list[dict]:
        """Find applicable patterns from *source_domain* that could help in *target_domain*."""
        analogies: list[dict] = []

        # Strategy 1: tag overlap — solutions in source whose tags appear
        # in the task description (simple keyword matching).
        task_lower = task.lower()
        for sol in self._solutions:
            if sol.domain != source_domain:
                continue
            overlap = [t for t in sol.tags if t.lower() in task_lower]
            if overlap or _text_overlap(sol.task_description, task) > 0.3:
                analogies.append(
                    {
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "source_task": sol.task_description,
                        "matching_tags": overlap,
                        "relevance": sol.success_score * max(len(overlap) * 0.25, 0.3),
                        "solution_summary": sol.solution,
                    }
                )

        # Strategy 2: if a knowledge_query callable exists, ask it.
        if self._knowledge_query is not None:
            try:
                extra = self._knowledge_query(
                    task=task,
                    source_domain=source_domain,
                    target_domain=target_domain,
                )
                if isinstance(extra, list):
                    analogies.extend(extra)
            except Exception:
                log.warning("knowledge_query failed during analogy search", exc_info=True)

        # Sort by relevance descending
        analogies.sort(key=lambda a: a.get("relevance", 0), reverse=True)
        return analogies

    def transfer_solution(
        self, solution: dict, from_domain: str, to_domain: str
    ) -> dict:
        """Adapt a solution from one domain to another.

        The adaptation is intentionally lightweight: it relabels the domain
        and marks parameters that likely need re-tuning.
        """
        adapted = dict(solution)
        adapted["_original_domain"] = from_domain
        adapted["_target_domain"] = to_domain
        adapted["_transferred"] = True

        # Flag numeric parameters as needing re-calibration
        needs_tuning: list[str] = []
        for key, value in solution.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                needs_tuning.append(key)
        adapted["_needs_tuning"] = needs_tuning

        return adapted

    def get_cross_domain_insights(self) -> list[dict]:
        """Find patterns that appear successfully across multiple domains."""
        # Group solutions by their tag sets
        tag_domains: dict[str, set[str]] = {}
        tag_solutions: dict[str, list[_DomainSolution]] = {}
        for sol in self._solutions:
            for tag in sol.tags:
                tag_domains.setdefault(tag, set()).add(sol.domain)
                tag_solutions.setdefault(tag, []).append(sol)

        insights: list[dict] = []
        for tag, domains in tag_domains.items():
            if len(domains) >= 2:
                avg_score = (
                    sum(s.success_score for s in tag_solutions[tag])
                    / len(tag_solutions[tag])
                )
                insights.append(
                    {
                        "pattern": tag,
                        "domains": sorted(domains),
                        "occurrences": len(tag_solutions[tag]),
                        "avg_success_score": round(avg_score, 4),
                    }
                )

        insights.sort(key=lambda i: i["occurrences"], reverse=True)
        return insights


# ── helpers ─────────────────────────────────────────────────────────────

def _text_overlap(a: str, b: str) -> float:
    """Jaccard similarity of word sets — quick & cheap."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)
