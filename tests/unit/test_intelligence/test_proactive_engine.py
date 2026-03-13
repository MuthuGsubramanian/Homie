import time
from unittest.mock import MagicMock, patch

from homie_core.intelligence.proactive_engine import (
    ProactiveEngine,
    _DEFAULT_INTERVAL,
    _FLOW_INTERVAL,
)
from homie_core.intelligence.action_pipeline import ActionPipeline, PipelineConfig
from homie_core.intelligence.interruption_model import InterruptionModel
from homie_core.memory.working import WorkingMemory


def _make_engine(**kwargs):
    wm = kwargs.pop("working_memory", WorkingMemory())
    return ProactiveEngine(working_memory=wm, **kwargs), wm


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_proactive_engine_init():
    engine, wm = _make_engine()
    assert engine.suggestion_count == 0
    assert engine.get_staged_suggestions() == []


def test_proactive_engine_custom_intervals():
    engine, _ = _make_engine(base_interval=90.0, flow_interval=240.0)
    assert engine._base_interval == 90.0
    assert engine._flow_interval == 240.0


def test_min_interval_clamped():
    engine, _ = _make_engine(base_interval=5.0)
    assert engine._base_interval == 60.0  # clamped to _MIN_INTERVAL


# ------------------------------------------------------------------
# Interval adaptation
# ------------------------------------------------------------------

def test_interval_increases_in_flow():
    engine, wm = _make_engine()
    wm.update("in_flow", False)
    wm.update("flow_score", 0.0)
    base = engine._current_interval()

    wm.update("in_flow", True)
    wm.update("flow_score", 0.9)
    flow = engine._current_interval()

    assert flow > base


def test_interval_is_flow_interval_when_in_flow():
    engine, wm = _make_engine()
    wm.update("in_flow", True)
    assert engine._current_interval() == _FLOW_INTERVAL


# ------------------------------------------------------------------
# tick() respects interval
# ------------------------------------------------------------------

def test_tick_skips_if_too_early():
    engine, wm = _make_engine()
    # First tick should always run (last_run == 0)
    engine._last_run = time.time()  # pretend we just ran
    result = engine.tick()
    assert result == []


def test_tick_runs_after_interval():
    engine, wm = _make_engine(base_interval=60.0)
    # Set last_run far in the past
    engine._last_run = time.time() - 200
    wm.update("in_flow", False)
    wm.update("flow_score", 0.0)
    # Even with empty context, tick should run (returns empty suggestions)
    result = engine.tick()
    assert isinstance(result, list)


# ------------------------------------------------------------------
# Suggestion generation + staging
# ------------------------------------------------------------------

def test_suggestions_staged_in_working_memory():
    wm = WorkingMemory()
    wm.update("in_flow", False)
    wm.update("flow_score", 0.2)
    wm.update("minutes_in_task", 150.0)
    wm.update("activity_type", "coding")

    engine = ProactiveEngine(
        working_memory=wm,
        interruption_model=InterruptionModel(threshold=0.0),  # always interrupt
    )
    engine._last_run = 0  # force run
    result = engine.tick()

    # Should have produced at least one suggestion (break)
    assert len(result) > 0
    staged = wm.get("staged_suggestions", [])
    assert len(staged) > 0
    assert staged[0]["type"] in ("break", "help")


def test_interruption_model_gates_suggestions():
    wm = WorkingMemory()
    wm.update("in_flow", False)
    wm.update("flow_score", 0.2)
    wm.update("minutes_in_task", 150.0)
    wm.update("activity_type", "coding")

    # Threshold=1.0 means never interrupt
    engine = ProactiveEngine(
        working_memory=wm,
        interruption_model=InterruptionModel(threshold=1.0),
    )
    engine._last_run = 0
    result = engine.tick()
    assert result == []
    assert wm.get("staged_suggestions", []) == []


def test_empty_context_produces_no_suggestions():
    engine, wm = _make_engine()
    engine._last_run = 0
    result = engine.tick()
    assert result == []


# ------------------------------------------------------------------
# consume_staged_suggestions
# ------------------------------------------------------------------

def test_consume_clears_staged():
    wm = WorkingMemory()
    wm.update("staged_suggestions", [{"id": "s1", "type": "break"}])
    engine = ProactiveEngine(working_memory=wm)

    consumed = engine.consume_staged_suggestions()
    assert len(consumed) == 1
    assert wm.get("staged_suggestions") == []


# ------------------------------------------------------------------
# Feedback
# ------------------------------------------------------------------

def test_record_feedback_updates_pipeline():
    engine, wm = _make_engine()
    engine.record_feedback(
        suggestion_id="sug_001",
        suggestion_type="break",
        accepted=True,
        minutes_in_task=60.0,
    )
    summary = engine._pipeline.get_feedback_summary()
    assert summary["total"] == 1


# ------------------------------------------------------------------
# Anomaly detector integration
# ------------------------------------------------------------------

def test_anomaly_detector_fed_during_cycle():
    from homie_core.intelligence.anomaly_detector import AnomalyDetector
    ad = AnomalyDetector(n_trees=5, sample_size=8, seed=42)

    wm = WorkingMemory()
    wm.update("in_flow", False)
    wm.update("flow_score", 0.3)
    wm.update("minutes_in_task", 30.0)
    wm.update("switch_count_30m", 10)

    engine = ProactiveEngine(
        working_memory=wm,
        anomaly_detector=ad,
        interruption_model=InterruptionModel(threshold=0.0),
    )
    engine._last_run = 0
    engine.tick()
    # The anomaly detector should have received a data point
    assert len(ad._buffer) >= 1


# ------------------------------------------------------------------
# Stuck task detection
# ------------------------------------------------------------------

def test_stuck_task_suggestion():
    wm = WorkingMemory()
    wm.update("in_flow", False)
    wm.update("flow_score", 0.2)
    wm.update("minutes_in_task", 60.0)
    wm.update("activity_type", "debugging")
    wm.update("task_description", "fix auth bug")

    engine = ProactiveEngine(
        working_memory=wm,
        interruption_model=InterruptionModel(threshold=0.0),
    )
    engine._last_run = 0
    result = engine.tick()

    types = [s["type"] for s in result]
    assert "help" in types


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------

def test_serialize_roundtrip():
    wm = WorkingMemory()
    engine = ProactiveEngine(working_memory=wm)
    engine.record_feedback("s1", "break", accepted=True)

    data = engine.serialize()
    engine2 = ProactiveEngine.deserialize(data, working_memory=wm)
    assert engine2._pipeline.get_feedback_summary()["total"] == 1


# ------------------------------------------------------------------
# Max 5 staged suggestions
# ------------------------------------------------------------------

def test_staged_suggestions_capped_at_5():
    wm = WorkingMemory()
    # Pre-stage 4 suggestions
    wm.update("staged_suggestions", [{"id": f"s{i}"} for i in range(4)])
    wm.update("in_flow", False)
    wm.update("flow_score", 0.2)
    wm.update("minutes_in_task", 150.0)
    wm.update("activity_type", "coding")

    engine = ProactiveEngine(
        working_memory=wm,
        interruption_model=InterruptionModel(threshold=0.0),
    )
    engine._last_run = 0
    engine.tick()

    staged = wm.get("staged_suggestions", [])
    assert len(staged) <= 5


# ------------------------------------------------------------------
# Integration with observer loop
# ------------------------------------------------------------------

def test_observer_loop_accepts_proactive_engine():
    from homie_core.intelligence.observer_loop import ObserverLoop
    from homie_core.intelligence.task_graph import TaskGraph

    wm = WorkingMemory()
    tg = TaskGraph()
    pe = ProactiveEngine(working_memory=wm)

    loop = ObserverLoop(
        working_memory=wm,
        task_graph=tg,
        proactive_engine=pe,
    )
    assert loop._proactive is pe
    assert not loop.is_running
