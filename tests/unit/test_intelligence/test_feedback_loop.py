import time
from homie_core.intelligence.feedback_loop import FeedbackLoop, FeedbackEvent, FeedbackType


def test_record_acceptance():
    fl = FeedbackLoop()
    fl.record(FeedbackEvent(
        suggestion_id="sug_001",
        suggestion_type="break",
        feedback_type=FeedbackType.ACCEPTED,
    ))
    assert fl.total_feedback == 1
    assert fl.acceptance_rate("break") == 1.0


def test_record_dismissal():
    fl = FeedbackLoop()
    fl.record(FeedbackEvent(
        suggestion_id="sug_001",
        suggestion_type="break",
        feedback_type=FeedbackType.DISMISSED,
    ))
    assert fl.acceptance_rate("break") == 0.0


def test_mixed_feedback():
    fl = FeedbackLoop()
    fl.record(FeedbackEvent("s1", "break", FeedbackType.ACCEPTED))
    fl.record(FeedbackEvent("s2", "break", FeedbackType.ACCEPTED))
    fl.record(FeedbackEvent("s3", "break", FeedbackType.DISMISSED))
    rate = fl.acceptance_rate("break")
    assert abs(rate - 2/3) < 0.01


def test_no_feedback_returns_neutral():
    fl = FeedbackLoop()
    assert fl.acceptance_rate("workflow") == 0.5


def test_get_feedback_summary():
    fl = FeedbackLoop()
    fl.record(FeedbackEvent("s1", "break", FeedbackType.ACCEPTED))
    fl.record(FeedbackEvent("s2", "workflow", FeedbackType.DISMISSED))
    fl.record(FeedbackEvent("s3", "break", FeedbackType.SNOOZED))
    summary = fl.get_summary()
    assert summary["total"] == 3
    assert "break" in summary["by_type"]
    assert "workflow" in summary["by_type"]


def test_get_recent_feedback():
    fl = FeedbackLoop()
    fl.record(FeedbackEvent("s1", "break", FeedbackType.ACCEPTED))
    fl.record(FeedbackEvent("s2", "workflow", FeedbackType.DISMISSED))
    fl.record(FeedbackEvent("s3", "anomaly", FeedbackType.ACCEPTED))
    recent = fl.get_recent(n=2)
    assert len(recent) == 2
    assert recent[0].suggestion_id == "s3"  # most recent first


def test_feedback_with_reason():
    fl = FeedbackLoop()
    fl.record(FeedbackEvent(
        suggestion_id="s1",
        suggestion_type="break",
        feedback_type=FeedbackType.DISMISSED,
        reason="I'm in flow right now",
    ))
    recent = fl.get_recent(n=1)
    assert recent[0].reason == "I'm in flow right now"


def test_learning_signals():
    fl = FeedbackLoop()
    fl.record(FeedbackEvent("s1", "break", FeedbackType.ACCEPTED))
    fl.record(FeedbackEvent("s2", "break", FeedbackType.DISMISSED))
    fl.record(FeedbackEvent("s3", "workflow", FeedbackType.ACCEPTED))
    signals = fl.get_learning_signals()
    assert "break" in signals
    assert "workflow" in signals
    # Each signal has accepted_count and dismissed_count
    assert signals["break"]["accepted"] == 1
    assert signals["break"]["dismissed"] == 1


def test_serialize_deserialize():
    fl = FeedbackLoop()
    fl.record(FeedbackEvent("s1", "break", FeedbackType.ACCEPTED))
    data = fl.serialize()
    fl2 = FeedbackLoop.deserialize(data)
    assert fl2.total_feedback == 1
    assert fl2.acceptance_rate("break") == 1.0


def test_feedback_type_snoozed():
    fl = FeedbackLoop()
    fl.record(FeedbackEvent("s1", "break", FeedbackType.SNOOZED))
    # Snoozed counts as neither accepted nor dismissed
    assert fl.acceptance_rate("break") == 0.5  # neutral
    signals = fl.get_learning_signals()
    assert signals["break"]["snoozed"] == 1
