"""End-to-end action pipeline orchestrating suggestion generation to delivery.

Connects: context -> suggestion engine -> ranker -> explanation -> delivery -> feedback.
This is the main entry point for Homie's proactive intelligence.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from homie_core.intelligence.suggestion_engine import SuggestionEngine, Suggestion
from homie_core.intelligence.suggestion_ranker import SuggestionRanker
from homie_core.intelligence.explanation_chain import ExplanationChain
from homie_core.intelligence.feedback_loop import FeedbackLoop, FeedbackEvent, FeedbackType


@dataclass
class PipelineConfig:
    max_suggestions: int = 3
    cooldown_seconds: float = 300
    min_confidence: float = 0.3


class ActionPipeline:
    """Orchestrates the full suggestion-to-feedback pipeline.

    1. Generate suggestions from context (SuggestionEngine)
    2. Rank them using Thompson sampling (SuggestionRanker)
    3. Attach explanations (ExplanationChain)
    4. Return top-N for delivery
    5. Collect feedback and update models
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()
        self._engine = SuggestionEngine(
            cooldown_seconds=self._config.cooldown_seconds
        )
        self._ranker = SuggestionRanker()
        self._feedback = FeedbackLoop()

    def process(self, context: dict) -> dict:
        """Process context through the full pipeline.

        Returns dict with:
        - suggestions: list of ranked suggestions with explanations
        - context_summary: brief context description
        """
        # 1. Generate raw suggestions
        raw_suggestions = self._engine.generate(context)

        if not raw_suggestions:
            return {"suggestions": [], "context_summary": self._summarize_context(context)}

        # 2. Filter by minimum confidence
        filtered = [s for s in raw_suggestions if s.confidence >= self._config.min_confidence]

        # 3. Rank using Thompson sampling
        rank_items = [
            {"id": s.id, "type": s.type.value, "confidence": s.confidence, "_suggestion": s}
            for s in filtered
        ]
        ranked = self._ranker.rank(rank_items, top_n=self._config.max_suggestions)

        # 4. Build explanations
        result_suggestions = []
        for item in ranked:
            sug: Suggestion = item["_suggestion"]
            chain = ExplanationChain.from_evidence(
                source=sug.source,
                conclusion=sug.body,
                evidence=sug.evidence,
                confidence=sug.confidence,
            )
            result_suggestions.append({
                "id": sug.id,
                "type": sug.type.value,
                "title": sug.title,
                "body": sug.body,
                "confidence": sug.confidence,
                "explanation": {
                    "short": chain.explain_short(),
                    "detailed": chain.explain_detailed(),
                    "sources": chain.get_sources(),
                },
            })

        return {
            "suggestions": result_suggestions,
            "context_summary": self._summarize_context(context),
        }

    def record_feedback(
        self,
        suggestion_id: str,
        suggestion_type: str,
        accepted: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Record user feedback on a suggestion."""
        ftype = FeedbackType.ACCEPTED if accepted else FeedbackType.DISMISSED
        self._feedback.record(FeedbackEvent(
            suggestion_id=suggestion_id,
            suggestion_type=suggestion_type,
            feedback_type=ftype,
            reason=reason,
        ))
        # Update ranker with outcome
        self._ranker.record_outcome(suggestion_type, accepted)

    def get_feedback_summary(self) -> dict:
        return self._feedback.get_summary()

    def get_intelligence_report(self, context: dict) -> dict:
        """Generate a human-readable intelligence report from context."""
        return {
            "activity": context.get("activity_type", "unknown"),
            "flow": {
                "score": context.get("flow_score", 0.5),
                "in_flow": context.get("in_flow", False),
            },
            "minutes_in_task": context.get("minutes_in_task", 0),
            "anomaly_score": context.get("anomaly_score", 0),
        }

    def _summarize_context(self, context: dict) -> str:
        activity = context.get("activity_type", "unknown")
        flow = context.get("flow_score", 0.5)
        mins = context.get("minutes_in_task", 0)
        parts = [f"Activity: {activity}"]
        if mins > 0:
            parts.append(f"Duration: {int(mins)}m")
        parts.append(f"Focus: {flow:.0%}")
        return " | ".join(parts)

    def serialize(self) -> dict:
        return {
            "ranker": self._ranker.serialize(),
            "feedback": self._feedback.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> ActionPipeline:
        pipeline = cls()
        if "ranker" in data:
            pipeline._ranker = SuggestionRanker.deserialize(data["ranker"])
        if "feedback" in data:
            pipeline._feedback = FeedbackLoop.deserialize(data["feedback"])
        return pipeline
