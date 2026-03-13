import pytest
from homie_core.feedback.reflection import ReflectionEngine
from homie_core.feedback.beliefs import BeliefSystem
from homie_core.feedback.patterns import PatternDetector
from homie_core.feedback.collector import FeedbackCollector
from homie_core.storage.database import Database


@pytest.fixture
def reflection(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    bs = BeliefSystem(db=db)
    pd = PatternDetector(db=db)
    col = FeedbackCollector(db=db)
    return ReflectionEngine(db=db, belief_system=bs, pattern_detector=pd), bs, col


def test_generate_reflection_empty(reflection):
    ref, _, _ = reflection
    result = ref.generate_reflection()
    assert "total_beliefs" in result
    assert result["total_beliefs"] == 0


def test_generate_reflection_with_data(reflection):
    ref, bs, col = reflection
    bs.add_belief("user likes Python", confidence=0.9)
    bs.add_belief("stale belief", confidence=0.15)
    for _ in range(3):
        col.record_correction("tabs", "spaces")
    result = ref.generate_reflection()
    assert result["total_beliefs"] == 2
    assert result["high_confidence_beliefs"] == 1
    assert len(result["insights"]) > 0


def test_acceptance_trend(reflection):
    ref, _, col = reflection
    col.record_preference("action1", accepted=True)
    col.record_preference("action2", accepted=True)
    col.record_preference("action3", accepted=False)
    trend = ref.get_acceptance_trend()
    assert 0.6 <= trend <= 0.7
