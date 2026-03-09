import pytest
from homie_core.storage.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    yield d
    d.close()


def test_initialize_creates_tables(db):
    tables = db.list_tables()
    assert "semantic_memory" in tables
    assert "beliefs" in tables
    assert "profile" in tables
    assert "feedback" in tables
    assert "episodes_meta" in tables


def test_store_and_retrieve_fact(db):
    db.store_fact("user prefers Python", confidence=0.9, tags=["work"])
    facts = db.get_facts(min_confidence=0.5)
    assert len(facts) == 1
    assert facts[0]["fact"] == "user prefers Python"
    assert facts[0]["confidence"] == 0.9


def test_store_belief(db):
    db.store_belief("likes concise responses", confidence=0.85, source_count=10, context_tags=["communication"])
    beliefs = db.get_beliefs()
    assert len(beliefs) == 1
    assert beliefs[0]["belief"] == "likes concise responses"


def test_record_feedback(db):
    db.record_feedback(channel="correction", content="prefers spaces over tabs", context={"app": "vscode"})
    feedback = db.get_recent_feedback(limit=10)
    assert len(feedback) == 1
    assert feedback[0]["channel"] == "correction"


def test_store_profile_domain(db):
    db.store_profile("music", {"top_genres": ["electronic", "lo-fi"]})
    profile = db.get_profile("music")
    assert profile["top_genres"] == ["electronic", "lo-fi"]


def test_update_profile_domain(db):
    db.store_profile("music", {"top_genres": ["electronic"]})
    db.store_profile("music", {"top_genres": ["electronic", "lo-fi"]})
    profile = db.get_profile("music")
    assert len(profile["top_genres"]) == 2
