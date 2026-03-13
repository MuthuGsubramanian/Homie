import pytest
from homie_core.memory.semantic import SemanticMemory
from homie_core.storage.database import Database


@pytest.fixture
def semantic(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    return SemanticMemory(db=db)


def test_learn_fact(semantic):
    semantic.learn("User prefers Python", confidence=0.9, tags=["work"])
    facts = semantic.get_facts(min_confidence=0.5)
    assert len(facts) == 1
    assert facts[0]["fact"] == "User prefers Python"


def test_reinforce_increases_confidence(semantic):
    semantic.learn("User likes dark mode", confidence=0.6, tags=["preferences"])
    semantic.reinforce("User likes dark mode", boost=0.1)
    facts = semantic.get_facts()
    matching = [f for f in facts if f["fact"] == "User likes dark mode"]
    assert matching[0]["confidence"] > 0.6


def test_forget_by_topic(semantic):
    semantic.learn("User likes rock music", confidence=0.8, tags=["music"])
    semantic.learn("User prefers Python", confidence=0.9, tags=["work"])
    semantic.forget_topic("music")
    facts = semantic.get_facts()
    assert all("music" not in str(f["tags"]) for f in facts)


def test_get_profile_summary(semantic):
    semantic.set_profile("work", {"role": "software engineer", "languages": ["Python"]})
    profile = semantic.get_profile("work")
    assert profile["role"] == "software engineer"
