import pytest
from datetime import timedelta
from homie_core.memory.forgetting import ForgettingCurve
from homie_core.storage.database import Database
from homie_core.utils import utc_now


@pytest.fixture
def curve(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    return ForgettingCurve(db=db, decay_rate=0.1)


def test_recent_high_access_stays_relevant(curve):
    score = curve.calculate_relevance(0.9, utc_now().isoformat(), access_count=10)
    assert score > 0.5


def test_old_low_access_decays(curve):
    old = (utc_now() - timedelta(days=60)).isoformat()
    score = curve.calculate_relevance(0.5, old, access_count=1)
    assert score < 0.1


def test_decay_archives_old_facts(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    old_time = (utc_now() - timedelta(days=365)).isoformat()
    db._conn.execute(
        "INSERT INTO semantic_memory (fact, confidence, source_count, tags, created_at, last_confirmed) VALUES (?, ?, ?, ?, ?, ?)",
        ("old fact", 0.3, 1, "[]", old_time, old_time),
    )
    db._conn.commit()
    curve = ForgettingCurve(db=db, decay_rate=0.1)
    archived = curve.decay_all(threshold=0.05)
    assert archived >= 1
