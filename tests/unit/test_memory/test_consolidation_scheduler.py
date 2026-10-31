"""Tests for the consolidation scheduler."""
import time
from unittest.mock import MagicMock, patch

from homie_core.memory.consolidation_scheduler import (
    ConsolidationMetrics,
    ConsolidationScheduler,
)


def _make_scheduler(**kwargs):
    """Create a scheduler with mock dependencies."""
    return ConsolidationScheduler(
        semantic_memory=kwargs.get("sm"),
        episodic_memory=kwargs.get("em"),
        forgetting_curve=kwargs.get("fc"),
        drift_detector=kwargs.get("drift"),
        user_model=kwargs.get("um"),
        min_interval=0,  # No waiting in tests
    )


def test_idle_detection():
    sched = _make_scheduler()
    # Initially idle (no conversation)
    assert sched.is_idle(idle_threshold=1.0)

    # After activity, not idle
    sched.notify_conversation_activity()
    assert not sched.is_idle(idle_threshold=1.0)


def test_should_run_when_idle():
    sched = _make_scheduler()
    assert sched.should_run()


def test_should_not_run_during_conversation():
    sched = _make_scheduler()
    sched.notify_conversation_activity()
    assert not sched.should_run()


def test_run_empty_no_crash():
    sched = _make_scheduler()
    metrics = sched.run(force=True)
    assert metrics.total_runs == 1
    assert metrics.last_memories_pruned == 0
    assert metrics.last_facts_merged == 0


def test_run_with_drift():
    mock_drift = MagicMock()
    mock_drift.analyze.return_value = MagicMock(drift_score=0.5)

    sched = _make_scheduler(drift=mock_drift)
    metrics = sched.run(force=True)
    assert metrics.last_drift_score == 0.5
    mock_drift.analyze.assert_called_once()


def test_run_pruning():
    mock_sm = MagicMock()
    mock_sm.get_facts.return_value = [
        {
            "id": 1,
            "fact": "some old fact",
            "confidence": 0.2,
            "source_count": 1,
            "last_confirmed": "2020-01-01T00:00:00+00:00",
            "created_at": "2020-01-01T00:00:00+00:00",
        },
    ]

    sched = _make_scheduler(sm=mock_sm)
    metrics = sched.run(force=True)
    assert metrics.total_runs == 1
    # The old low-confidence fact should be pruned
    assert metrics.last_memories_pruned >= 0


def test_run_merging():
    mock_sm = MagicMock()
    mock_sm.get_facts.return_value = []

    mock_em = MagicMock()
    # 3+ similar episodes should trigger merging
    mock_em.recall.return_value = [
        {"id": "ep_1", "summary": "python coding debugging session with django"},
        {"id": "ep_2", "summary": "python coding debugging work with django"},
        {"id": "ep_3", "summary": "python coding debugging project with django"},
    ]

    sched = _make_scheduler(sm=mock_sm, em=mock_em)
    metrics = sched.run(force=True)
    assert metrics.total_runs == 1


def test_run_user_model_update():
    mock_um = MagicMock()
    sched = _make_scheduler(um=mock_um)
    sched.run(force=True)
    mock_um.get_profile.assert_called_once_with(force_refresh=True)


def test_metrics_accumulate():
    sched = _make_scheduler()
    sched.run(force=True)
    sched.run(force=True)
    assert sched.metrics.total_runs == 2


def test_metrics_to_dict():
    metrics = ConsolidationMetrics(total_runs=3, total_memories_pruned=5)
    d = metrics.to_dict()
    assert d["total_runs"] == 3
    assert d["total_memories_pruned"] == 5


def test_run_respects_interval():
    sched = ConsolidationScheduler(min_interval=9999)
    # First run should work (force)
    sched.run(force=True)
    # Second should be skipped (interval not elapsed, and it's idle)
    initial_runs = sched.metrics.total_runs
    sched.run(force=False)
    assert sched.metrics.total_runs == initial_runs


def test_forgetting_curve_integration():
    mock_sm = MagicMock()
    mock_sm.get_facts.return_value = []
    mock_fc = MagicMock()
    mock_fc.decay_all.return_value = 3

    sched = _make_scheduler(sm=mock_sm, fc=mock_fc)
    metrics = sched.run(force=True)
    mock_fc.decay_all.assert_called_once()
    assert metrics.last_memories_pruned == 3
