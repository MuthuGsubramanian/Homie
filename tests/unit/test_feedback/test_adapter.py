import pytest
from homie_core.feedback.adapter import BehavioralAdapter
from homie_core.feedback.beliefs import BeliefSystem
from homie_core.feedback.patterns import PatternDetector
from homie_core.storage.database import Database


@pytest.fixture
def adapter(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    bs = BeliefSystem(db=db)
    pd = PatternDetector(db=db)
    return BehavioralAdapter(belief_system=bs, pattern_detector=pd), bs


def test_should_suggest_default_true(adapter):
    adp, _ = adapter
    assert adp.should_suggest("new action") is True


def test_should_not_suggest_low_confidence(adapter):
    adp, bs = adapter
    bs.add_belief("break reminder", confidence=0.3)
    assert adp.should_suggest("break reminder") is False


def test_get_suggestion_timing_default(adapter):
    adp, _ = adapter
    assert adp.get_suggestion_timing() == "immediate"


def test_adjust_threshold(adapter):
    adp, _ = adapter
    original = adp._suggestion_threshold
    adp.adjust_threshold("fewer_suggestions")
    assert adp._suggestion_threshold > original
    adp.adjust_threshold("more_suggestions")
    adp.adjust_threshold("more_suggestions")
    assert adp._suggestion_threshold < original
