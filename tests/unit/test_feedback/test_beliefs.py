import pytest
from homie_core.feedback.beliefs import BeliefSystem
from homie_core.storage.database import Database


@pytest.fixture
def beliefs(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    return BeliefSystem(db=db)


def test_add_and_get_belief(beliefs):
    bid = beliefs.add_belief("user prefers concise responses", confidence=0.8, context_tags=["communication"])
    all_beliefs = beliefs.get_beliefs()
    assert len(all_beliefs) == 1
    assert all_beliefs[0]["belief"] == "user prefers concise responses"


def test_reinforce_belief(beliefs):
    bid = beliefs.add_belief("user likes dark mode", confidence=0.6)
    beliefs.reinforce(bid, boost=0.15)
    updated = beliefs.get_beliefs()
    assert updated[0]["confidence"] == pytest.approx(0.75, abs=0.01)


def test_weaken_belief(beliefs):
    bid = beliefs.add_belief("user likes pop music", confidence=0.7)
    beliefs.weaken(bid, penalty=0.2)
    updated = beliefs.get_beliefs()
    assert updated[0]["confidence"] == pytest.approx(0.5, abs=0.01)


def test_decay_all(beliefs):
    beliefs.add_belief("belief 1", confidence=0.5)
    beliefs.add_belief("belief 2", confidence=0.3)
    beliefs.decay_all(rate=0.05)
    all_b = beliefs.get_beliefs()
    assert all_b[0]["confidence"] < 0.5
    assert all_b[1]["confidence"] < 0.3


def test_find_belief(beliefs):
    beliefs.add_belief("user prefers Python", confidence=0.9)
    beliefs.add_belief("user likes dark mode", confidence=0.7)
    results = beliefs.find_belief("python")
    assert len(results) == 1
    assert "Python" in results[0]["belief"]
