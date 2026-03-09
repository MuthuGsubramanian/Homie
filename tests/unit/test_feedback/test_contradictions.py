import pytest
from homie_core.feedback.beliefs import BeliefSystem
from homie_core.feedback.contradictions import ContradictionResolver
from homie_core.storage.database import Database


@pytest.fixture
def resolver(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    bs = BeliefSystem(db=db)
    return ContradictionResolver(belief_system=bs), bs


def test_detect_contradiction(resolver):
    res, bs = resolver
    bs.add_belief("user likes dark mode", confidence=0.8)
    contradictions = res.detect_contradictions("user dislikes dark mode")
    assert len(contradictions) >= 1


def test_no_contradiction(resolver):
    res, bs = resolver
    bs.add_belief("user likes Python", confidence=0.8)
    contradictions = res.detect_contradictions("user likes FastAPI")
    assert len(contradictions) == 0


def test_resolve_replace(resolver):
    res, bs = resolver
    bid = bs.add_belief("user likes tabs", confidence=0.8)
    result = res.resolve(bid, "user likes spaces", strategy="replace")
    assert result["action"] == "replaced"


def test_resolve_context_split(resolver):
    res, bs = resolver
    bid = bs.add_belief("user likes quiet", confidence=0.7)
    result = res.resolve(bid, "user likes music while coding", strategy="context_split", new_context=["work"])
    assert result["action"] == "context_split"
