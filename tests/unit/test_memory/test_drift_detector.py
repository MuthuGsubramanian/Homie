"""Tests for semantic drift detection."""
from unittest.mock import MagicMock

from homie_core.memory.drift_detector import (
    DriftReport,
    SemanticDriftDetector,
    _extract_topic_distribution,
    _jensen_shannon_divergence,
)


def test_extract_topic_distribution():
    summaries = [
        "5-turn coding session. Topics: python, debugging",
        "3-turn coding session. Topics: python, testing",
    ]
    dist = _extract_topic_distribution(summaries)
    assert dist["python"] == 2
    assert dist["coding"] == 2
    assert "the" not in dist  # stop word filtered


def test_jsd_identical_distributions():
    from collections import Counter
    p = Counter({"python": 5, "coding": 3})
    q = Counter({"python": 5, "coding": 3})
    assert _jensen_shannon_divergence(p, q) == 0.0


def test_jsd_different_distributions():
    from collections import Counter
    p = Counter({"python": 10, "coding": 10})
    q = Counter({"cooking": 10, "recipes": 10})
    score = _jensen_shannon_divergence(p, q)
    assert score > 0.8  # Very different distributions


def test_jsd_empty_distributions():
    from collections import Counter
    assert _jensen_shannon_divergence(Counter(), Counter()) == 0.0


def test_drift_detector_no_memory():
    detector = SemanticDriftDetector(episodic_memory=None)
    report = detector.analyze()
    assert report.drift_score == 0.0
    assert not report.is_significant


def test_drift_detector_insufficient_data():
    mock_em = MagicMock()
    mock_em.recall.return_value = [
        {"summary": "coding session"},
        {"summary": "another session"},
    ]
    detector = SemanticDriftDetector(episodic_memory=mock_em, recent_window=10)
    report = detector.analyze()
    assert report.drift_score == 0.0


def test_drift_detector_detects_shift():
    mock_em = MagicMock()
    # Recent episodes: cooking topics
    recent = [{"summary": f"cooking recipes meal preparation food"} for _ in range(10)]
    # Historical: coding topics
    historical = [{"summary": f"python coding debugging testing software"} for _ in range(20)]
    mock_em.recall.return_value = recent + historical

    detector = SemanticDriftDetector(episodic_memory=mock_em, recent_window=10)
    report = detector.analyze()
    assert report.drift_score > 0.3
    assert len(report.emerging_topics) > 0 or len(report.fading_topics) > 0


def test_drift_detector_stable_topics():
    mock_em = MagicMock()
    # Both windows have same topics
    episodes = [{"summary": "python coding debugging testing"} for _ in range(30)]
    mock_em.recall.return_value = episodes

    detector = SemanticDriftDetector(episodic_memory=mock_em, recent_window=10)
    report = detector.analyze()
    assert report.drift_score < 0.1
    assert not report.is_significant


def test_drift_trend():
    detector = SemanticDriftDetector(episodic_memory=None)
    # Multiple analyses will produce 0.0 scores since no memory
    detector.analyze()
    detector.analyze()
    trend = detector.get_drift_trend()
    assert len(trend) == 2
    assert all(s == 0.0 for s in trend)


def test_drift_report_significance():
    report = DriftReport(drift_score=0.5, emerging_topics=[], fading_topics=[], stable_topics=[])
    assert report.is_significant

    report2 = DriftReport(drift_score=0.2, emerging_topics=[], fading_topics=[], stable_topics=[])
    assert not report2.is_significant
