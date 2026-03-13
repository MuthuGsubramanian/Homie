"""Proactive suggestion engine — orchestrates suggestion generation and delivery.

Runs on a timer (adaptive based on flow state), gathers context from working
memory, generates suggestion candidates via ActionPipeline, gates them through
the InterruptionModel, and stages approved suggestions in working memory for
the overlay to pick up.

Design decisions:
- Suggestions are STAGED in working memory, never shown directly.
- The InterruptionModel gates ALL proactive interruptions.
- Flow state modulates the check interval (longer when in flow).
- Per-type cooldowns are handled by SuggestionEngine internally.
"""
from __future__ import annotations

import time
from typing import Optional

from homie_core.intelligence.action_pipeline import ActionPipeline, PipelineConfig
from homie_core.intelligence.interruption_model import InterruptionModel
from homie_core.intelligence.anomaly_detector import AnomalyDetector
from homie_core.memory.working import WorkingMemory


# Interval bounds (seconds)
_DEFAULT_INTERVAL = 120.0      # 2 minutes baseline
_FLOW_INTERVAL = 300.0         # 5 minutes when user is in flow
_MIN_INTERVAL = 60.0           # never faster than 1 minute
_MAX_INTERVAL = 600.0          # never slower than 10 minutes


class ProactiveEngine:
    """Orchestrator that periodically generates and stages proactive suggestions.

    This is NOT a thread — it exposes a ``tick()`` method that the observer
    loop calls every N poll cycles.  The engine decides internally whether
    enough time has elapsed to run a full suggestion cycle.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        interruption_model: Optional[InterruptionModel] = None,
        pipeline: Optional[ActionPipeline] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        base_interval: float = _DEFAULT_INTERVAL,
        flow_interval: float = _FLOW_INTERVAL,
    ):
        self._wm = working_memory
        self._interruption = interruption_model or InterruptionModel()
        self._pipeline = pipeline or ActionPipeline(
            config=PipelineConfig(cooldown_seconds=300)
        )
        self._anomaly = anomaly_detector
        self._base_interval = max(base_interval, _MIN_INTERVAL)
        self._flow_interval = max(flow_interval, _MIN_INTERVAL)
        self._last_run: float = 0.0
        self._suggestion_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> list[dict]:
        """Called by the observer loop on every poll cycle.

        Returns a (possibly empty) list of suggestion dicts that were
        staged in working memory during this tick.
        """
        now = time.time()
        interval = self._current_interval()
        if now - self._last_run < interval:
            return []

        self._last_run = now
        return self._run_cycle()

    def record_feedback(
        self,
        suggestion_id: str,
        suggestion_type: str,
        accepted: bool,
        reason: Optional[str] = None,
        minutes_in_task: float = 0.0,
        switch_freq: float = 0.0,
        minutes_since_interaction: float = 0.0,
    ) -> None:
        """Record user feedback and update all underlying models."""
        self._pipeline.record_feedback(
            suggestion_id=suggestion_id,
            suggestion_type=suggestion_type,
            accepted=accepted,
            reason=reason,
        )
        # Also train the interruption model
        self._interruption.record_feedback(
            accepted=accepted,
            minutes_in_task=minutes_in_task,
            switch_freq_10min=switch_freq,
            minutes_since_interaction=minutes_since_interaction,
            category="suggestion",
        )

    def get_staged_suggestions(self) -> list[dict]:
        """Retrieve currently staged suggestions from working memory."""
        return self._wm.get("staged_suggestions", [])

    def consume_staged_suggestions(self) -> list[dict]:
        """Retrieve and clear staged suggestions (for display)."""
        suggestions = self._wm.get("staged_suggestions", [])
        self._wm.update("staged_suggestions", [])
        return suggestions

    @property
    def suggestion_count(self) -> int:
        return self._suggestion_count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_interval(self) -> float:
        """Adaptive interval: longer when user is in flow."""
        in_flow = self._wm.get("in_flow", False)
        flow_score = self._wm.get("flow_score", 0.5)

        if in_flow:
            return min(self._flow_interval, _MAX_INTERVAL)

        # Linearly interpolate between base and flow interval based on flow score
        t = max(0.0, min(1.0, flow_score))
        interval = self._base_interval + t * (self._flow_interval - self._base_interval)
        return max(_MIN_INTERVAL, min(interval, _MAX_INTERVAL))

    def _gather_context(self) -> dict:
        """Pull all relevant signals from working memory into a context dict."""
        snapshot = self._wm.snapshot()
        ctx: dict = {}

        # Core signals used by SuggestionEngine
        ctx["in_flow"] = snapshot.get("in_flow", False)
        ctx["flow_score"] = snapshot.get("flow_score", 0.5)
        ctx["minutes_in_task"] = snapshot.get("minutes_in_task", 0.0)
        ctx["activity_type"] = snapshot.get("activity_type", "unknown")
        ctx["context_shift"] = snapshot.get("context_shift", False)
        ctx["previous_activity"] = snapshot.get("previous_activity", "unknown")

        # Workflow prediction
        predicted = snapshot.get("predicted_next_activity")
        if predicted:
            ctx["predicted_next"] = [(predicted, 0.6)]

        # Anomaly detection — feed behavioral features if detector available
        if self._anomaly:
            features = self._build_anomaly_features(snapshot)
            if features:
                self._anomaly.stream_update(features)
                score = self._anomaly.score(features)
                ctx["anomaly_score"] = score

        # Rhythm / optimal windows
        if snapshot.get("rhythmic_score") is not None:
            from datetime import datetime
            hour = datetime.now().hour
            ctx["current_hour"] = hour
            ctx["optimal_windows"] = [
                {"hour": hour, "predicted_score": snapshot["rhythmic_score"]}
            ]

        # Preference shifts
        shifts = snapshot.get("preference_shifts")
        if shifts:
            ctx["preference_shifts"] = shifts

        # Stuck task heuristic: long time on one task with low flow
        mins = ctx["minutes_in_task"]
        flow = ctx["flow_score"]
        if mins >= 45 and flow < 0.4:
            task_desc = snapshot.get("task_description", "current task")
            ctx["stuck_tasks"] = [
                {"description": task_desc, "duration_minutes": mins}
            ]

        return ctx

    def _build_anomaly_features(self, snapshot: dict) -> list[float]:
        """Build a feature vector for the anomaly detector."""
        mins = snapshot.get("minutes_in_task", 0.0)
        flow = snapshot.get("flow_score", 0.5)
        switches = snapshot.get("switch_count_30m", 0)
        return [
            min(mins / 120.0, 1.0),
            flow,
            min(switches / 30.0, 1.0),
        ]

    def _run_cycle(self) -> list[dict]:
        """Execute one full suggestion cycle."""
        ctx = self._gather_context()
        if not ctx:
            return []

        # Generate + rank + explain through the pipeline
        result = self._pipeline.process(ctx)
        suggestions = result.get("suggestions", [])
        if not suggestions:
            return []

        # Gate each suggestion through the interruption model
        mins = ctx.get("minutes_in_task", 0.0)
        switch_freq = self._wm.get("switch_count_30m", 0) / 3.0  # approx per 10 min
        mins_since = 0.0  # we don't track this yet; default to 0

        approved: list[dict] = []
        for sug in suggestions:
            category = sug.get("type", "suggestion")
            if self._interruption.should_interrupt(
                minutes_in_task=mins,
                switch_freq_10min=switch_freq,
                minutes_since_interaction=mins_since,
                category=category,
            ):
                sug["staged_at"] = time.time()
                approved.append(sug)

        if approved:
            # Stage in working memory (append to existing, keep max 5)
            existing = self._wm.get("staged_suggestions", [])
            merged = existing + approved
            self._wm.update("staged_suggestions", merged[-5:])
            self._suggestion_count += len(approved)

        return approved

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        return {
            "pipeline": self._pipeline.serialize(),
            "interruption": self._interruption.serialize(),
            "suggestion_count": self._suggestion_count,
        }

    @classmethod
    def deserialize(cls, data: dict, working_memory: WorkingMemory) -> ProactiveEngine:
        engine = cls(working_memory=working_memory)
        if "pipeline" in data:
            engine._pipeline = ActionPipeline.deserialize(data["pipeline"])
        if "interruption" in data:
            engine._interruption = InterruptionModel.deserialize(data["interruption"])
        engine._suggestion_count = data.get("suggestion_count", 0)
        return engine
