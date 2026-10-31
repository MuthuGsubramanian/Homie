"""Tests for memory importance scoring."""
from homie_core.memory.importance import (
    ImportanceScore,
    MemoryImportanceScorer,
    _detect_emotional_significance,
)
from homie_core.utils import utc_now


def test_detect_emotional_significance_high():
    assert _detect_emotional_significance("I love this amazing project") > 0.7


def test_detect_emotional_significance_low():
    assert _detect_emotional_significance("The code runs correctly") == 0.0


def test_detect_emotional_significance_empty():
    assert _detect_emotional_significance("") == 0.0


def test_detect_emotional_significance_life_event():
    assert _detect_emotional_significance("Got promoted at work today") > 0.7


def test_score_memory_basic():
    scorer = MemoryImportanceScorer()
    now = utc_now().isoformat()
    memory = {
        "id": 1,
        "fact": "User loves Python programming",
        "confidence": 0.8,
        "source_count": 5,
        "last_confirmed": now,
        "created_at": now,
    }
    score = scorer.score_memory(memory)
    assert isinstance(score, ImportanceScore)
    assert 0.0 <= score.total <= 1.0
    assert score.recency > 0.5  # Very recent
    assert score.emotional > 0.0  # "loves" is emotional


def test_score_memory_old_and_infrequent():
    scorer = MemoryImportanceScorer()
    memory = {
        "id": 2,
        "fact": "User mentioned a configuration option",
        "confidence": 0.3,
        "source_count": 1,
        "last_confirmed": "2020-01-01T00:00:00+00:00",
        "created_at": "2020-01-01T00:00:00+00:00",
    }
    score = scorer.score_memory(memory)
    assert score.recency < 0.1  # Very old
    assert score.frequency < 0.25  # Single access
    assert score.total < 0.3  # Low overall


def test_score_batch():
    scorer = MemoryImportanceScorer()
    now = utc_now().isoformat()
    memories = [
        {"id": 1, "fact": "User is excited about machine learning", "source_count": 10, "last_confirmed": now},
        {"id": 2, "fact": "User mentioned a config option", "source_count": 1, "last_confirmed": "2020-01-01T00:00:00+00:00"},
    ]
    scores = scorer.score_batch(memories)
    assert len(scores) == 2
    assert scores[0].total > scores[1].total


def test_score_connections():
    scorer = MemoryImportanceScorer()
    now = utc_now().isoformat()
    memories = [
        {"id": 1, "fact": "User works with Python and Django framework", "source_count": 1, "last_confirmed": now},
        {"id": 2, "fact": "User builds Django web applications with Python", "source_count": 1, "last_confirmed": now},
        {"id": 3, "fact": "User prefers Python Django for backend work", "source_count": 1, "last_confirmed": now},
    ]
    scores = scorer.score_batch(memories)
    # The first memory should have connections to the other two
    assert scores[0].connections > 0.0


def test_score_no_timestamp():
    scorer = MemoryImportanceScorer()
    memory = {"id": 1, "fact": "Some fact"}
    score = scorer.score_memory(memory)
    assert score.recency == 0.0
