from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import defaultdict


class SuggestionType(Enum):
    BREAK = "break"
    WORKFLOW = "workflow"
    ANOMALY = "anomaly"
    RHYTHM = "rhythm"
    INSIGHT = "insight"
    CONTEXT = "context"
    HELP = "help"


@dataclass
class Suggestion:
    type: SuggestionType
    title: str
    body: str
    confidence: float
    source: str
    evidence: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"sug_{int(time.time()*1000)}")


class SuggestionEngine:
    """Generates actionable suggestions from neural intelligence signals."""

    def __init__(self, cooldown_seconds: float = 300):
        self._cooldown = cooldown_seconds
        self._last_fired: dict[str, float] = defaultdict(float)  # type -> timestamp

    def _can_fire(self, stype: SuggestionType) -> bool:
        key = stype.value
        now = time.time()
        if now - self._last_fired[key] < self._cooldown:
            return False
        return True

    def _mark_fired(self, stype: SuggestionType) -> None:
        self._last_fired[stype.value] = time.time()

    def generate(self, context: dict) -> list[Suggestion]:
        if not context:
            return []
        suggestions = []
        # Run all generators
        for gen in [
            self._break_suggestion,
            self._workflow_suggestion,
            self._anomaly_suggestion,
            self._rhythm_suggestion,
            self._insight_suggestion,
            self._context_suggestion,
            self._stuck_suggestion,
        ]:
            result = gen(context)
            if result and self._can_fire(result.type):
                suggestions.append(result)
                self._mark_fired(result.type)
        return suggestions

    def _break_suggestion(self, ctx: dict) -> Optional[Suggestion]:
        if ctx.get("in_flow"):
            return None
        mins = ctx.get("minutes_in_task", 0)
        flow = ctx.get("flow_score", 0.5)
        if mins >= 90 and flow < 0.5:
            return Suggestion(
                type=SuggestionType.BREAK,
                title="Time for a break",
                body=f"You've been working for {int(mins)} minutes with declining focus.",
                confidence=min(0.9, 0.5 + (mins - 90) / 120),
                source="flow_detector",
                evidence={"minutes_in_task": mins, "flow_score": flow},
            )
        return None

    def _workflow_suggestion(self, ctx: dict) -> Optional[Suggestion]:
        predicted = ctx.get("predicted_next")
        if not predicted:
            return None
        top_action, top_prob = predicted[0]
        if top_prob >= 0.5:
            return Suggestion(
                type=SuggestionType.WORKFLOW,
                title=f"Next: {top_action}?",
                body=f"Based on your workflow pattern, you usually do '{top_action}' next.",
                confidence=top_prob,
                source="workflow_predictor",
                evidence={"predicted_next": predicted[:3]},
            )
        return None

    def _anomaly_suggestion(self, ctx: dict) -> Optional[Suggestion]:
        score = ctx.get("anomaly_score", 0)
        if score >= 0.7:
            return Suggestion(
                type=SuggestionType.ANOMALY,
                title="Unusual activity detected",
                body="Your current activity pattern differs from your usual routine.",
                confidence=score,
                source="anomaly_detector",
                evidence={"anomaly_score": score},
            )
        return None

    def _rhythm_suggestion(self, ctx: dict) -> Optional[Suggestion]:
        windows = ctx.get("optimal_windows", [])
        current_hour = ctx.get("current_hour")
        if not windows or current_hour is None:
            return None
        for w in windows:
            if abs(w["hour"] - current_hour) <= 1 and w["predicted_score"] > 0.7:
                return Suggestion(
                    type=SuggestionType.RHYTHM,
                    title="Peak productivity window",
                    body=f"This is one of your peak hours (score: {w['predicted_score']:.0%}). Great time for deep work!",
                    confidence=w["predicted_score"],
                    source="rhythm_model",
                    evidence={"optimal_window": w, "current_hour": current_hour},
                )
        return None

    def _insight_suggestion(self, ctx: dict) -> Optional[Suggestion]:
        shifts = ctx.get("preference_shifts", [])
        if not shifts:
            return None
        shift = shifts[0]
        return Suggestion(
            type=SuggestionType.INSIGHT,
            title=f"Preference shift: {shift['key']}",
            body=f"I noticed your preference for '{shift['key']}' in '{shift['domain']}' has changed.",
            confidence=0.7,
            source="preference_engine",
            evidence={"shift": shift},
        )

    def _context_suggestion(self, ctx: dict) -> Optional[Suggestion]:
        if not ctx.get("context_shift"):
            return None
        prev = ctx.get("previous_activity", "unknown")
        current = ctx.get("activity_type", "unknown")
        return Suggestion(
            type=SuggestionType.CONTEXT,
            title="Context switch detected",
            body=f"You switched from {prev} to {current}. Want me to pull up relevant context?",
            confidence=0.6,
            source="context_engine",
            evidence={"from": prev, "to": current},
        )

    def _stuck_suggestion(self, ctx: dict) -> Optional[Suggestion]:
        stuck = ctx.get("stuck_tasks", [])
        if not stuck:
            return None
        task = stuck[0]
        mins = task.get("duration_minutes", 0)
        return Suggestion(
            type=SuggestionType.HELP,
            title="Need help?",
            body=f"You seem stuck on a task ({mins} min). Want me to search for related resources?",
            confidence=min(0.9, 0.5 + mins / 60),
            source="task_graph",
            evidence={"stuck_task": task},
        )
