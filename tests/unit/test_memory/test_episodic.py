import pytest
from homie_core.memory.episodic import EpisodicMemory
from homie_core.storage.vectors import VectorStore
from homie_core.storage.database import Database


@pytest.fixture
def episodic(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    vs = VectorStore(tmp_path / "chroma")
    vs.initialize()
    em = EpisodicMemory(db=db, vector_store=vs)
    return em


def test_record_episode(episodic):
    eid = episodic.record(summary="User spent 2 hours debugging auth module", mood="frustrated", outcome="fixed", context_tags=["work", "coding"])
    assert eid is not None


def test_recall_by_query(episodic):
    episodic.record(summary="Debugged Python auth module", mood="frustrated", context_tags=["work"])
    episodic.record(summary="Listened to lo-fi music while coding", mood="relaxed", context_tags=["music"])
    results = episodic.recall("authentication debugging", n=1)
    assert len(results) == 1
    assert "auth" in results[0]["summary"]


def test_recall_returns_mood(episodic):
    episodic.record(summary="Had a great meeting", mood="happy", context_tags=["work"])
    results = episodic.recall("meeting", n=1)
    assert results[0]["mood"] == "happy"
