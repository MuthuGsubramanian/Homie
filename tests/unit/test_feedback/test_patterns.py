import pytest
from homie_core.feedback.patterns import PatternDetector
from homie_core.feedback.collector import FeedbackCollector
from homie_core.storage.database import Database


@pytest.fixture
def detector(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    return PatternDetector(db=db), FeedbackCollector(db=db)


def test_find_correction_clusters(detector):
    det, col = detector
    for _ in range(4):
        col.record_correction("tabs", "spaces")
    col.record_correction("semicolons", "none")
    clusters = det.find_correction_clusters(min_count=3)
    assert len(clusters) >= 1
    assert clusters[0]["count"] >= 3


def test_find_preference_patterns(detector):
    det, col = detector
    for _ in range(3):
        col.record_preference("break reminder", accepted=True)
    col.record_preference("break reminder", accepted=False)
    patterns = det.find_preference_patterns()
    assert "break reminder" in patterns
    assert patterns["break reminder"] == 0.75


def test_find_temporal_patterns(detector):
    det, col = detector
    for _ in range(5):
        col.record_teaching("test fact")
    patterns = det.find_temporal_patterns()
    assert isinstance(patterns, list)
