import random
from homie_core.intelligence.suggestion_ranker import SuggestionRanker


def test_rank_by_confidence():
    ranker = SuggestionRanker()
    items = [
        {"id": "a", "type": "break", "confidence": 0.9},
        {"id": "b", "type": "workflow", "confidence": 0.5},
        {"id": "c", "type": "anomaly", "confidence": 0.7},
    ]
    ranked = ranker.rank(items)
    # Highest confidence should generally be first (before learning)
    assert ranked[0]["id"] == "a"


def test_thompson_sampling_exploration():
    """Thompson sampling should occasionally explore lower-confidence items."""
    ranker = SuggestionRanker(seed=42)
    items = [
        {"id": "a", "type": "break", "confidence": 0.6},
        {"id": "b", "type": "workflow", "confidence": 0.55},
    ]
    # Run many rankings to verify both items appear first sometimes
    first_counts = {"a": 0, "b": 0}
    for i in range(100):
        ranker_i = SuggestionRanker(seed=i)
        ranked = ranker_i.rank(items)
        first_counts[ranked[0]["id"]] += 1
    assert first_counts["a"] > 0
    assert first_counts["b"] > 0


def test_record_outcome_updates_prior():
    ranker = SuggestionRanker()
    ranker.record_outcome("break", accepted=True)
    ranker.record_outcome("break", accepted=True)
    ranker.record_outcome("break", accepted=False)
    stats = ranker.get_type_stats("break")
    assert stats["alpha"] == 3  # 1 (prior) + 2 successes
    assert stats["beta"] == 2  # 1 (prior) + 1 failure


def test_learned_type_gets_boosted():
    ranker = SuggestionRanker(seed=0)
    # Train "break" to be very good
    for _ in range(50):
        ranker.record_outcome("break", accepted=True)
    # Train "anomaly" to be bad
    for _ in range(50):
        ranker.record_outcome("anomaly", accepted=False)
    items = [
        {"id": "a", "type": "break", "confidence": 0.5},
        {"id": "b", "type": "anomaly", "confidence": 0.8},
    ]
    # Over many trials, break should win most of the time despite lower confidence
    wins = sum(1 for _ in range(100) if SuggestionRanker(seed=_).rank(items, type_stats=ranker._stats)[0]["id"] == "a")
    assert wins > 60


def test_top_n():
    ranker = SuggestionRanker()
    items = [
        {"id": "a", "type": "break", "confidence": 0.9},
        {"id": "b", "type": "workflow", "confidence": 0.5},
        {"id": "c", "type": "anomaly", "confidence": 0.7},
    ]
    ranked = ranker.rank(items, top_n=2)
    assert len(ranked) == 2


def test_empty_input():
    ranker = SuggestionRanker()
    assert ranker.rank([]) == []


def test_serialize_deserialize():
    ranker = SuggestionRanker()
    ranker.record_outcome("break", True)
    ranker.record_outcome("workflow", False)
    data = ranker.serialize()
    ranker2 = SuggestionRanker.deserialize(data)
    assert ranker2.get_type_stats("break")["alpha"] == 2
    assert ranker2.get_type_stats("workflow")["beta"] == 2


def test_decay_over_time():
    ranker = SuggestionRanker(decay_rate=0.1)
    for _ in range(100):
        ranker.record_outcome("break", True)
    stats_before = ranker.get_type_stats("break")
    ranker.apply_decay()
    stats_after = ranker.get_type_stats("break")
    # Alpha should be reduced after decay
    assert stats_after["alpha"] < stats_before["alpha"]
