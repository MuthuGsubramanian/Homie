import pytest
from homie_core.feedback.collector import FeedbackCollector
from homie_core.storage.database import Database


@pytest.fixture
def collector(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    return FeedbackCollector(db=db)


def test_record_correction(collector):
    fid = collector.record_correction("tabs", "spaces", context={"app": "vscode"})
    assert fid > 0
    recent = collector.get_recent(channel="correction")
    assert len(recent) == 1
    assert "tabs" in recent[0]["content"]


def test_record_preference(collector):
    collector.record_preference("suggest break", accepted=True)
    recent = collector.get_recent(channel="preference")
    assert len(recent) == 1
    assert "Accepted" in recent[0]["content"]


def test_record_teaching(collector):
    collector.record_teaching("I am allergic to peanuts")
    recent = collector.get_recent(channel="teaching")
    assert len(recent) == 1


def test_record_satisfaction(collector):
    collector.record_satisfaction("code suggestion", "thumbs_up")
    recent = collector.get_recent(channel="satisfaction")
    assert "thumbs_up" in recent[0]["content"]


def test_record_onboarding(collector):
    collector.record_onboarding("What is your role?", "Software Engineer")
    recent = collector.get_recent(channel="onboarding")
    assert "Software Engineer" in recent[0]["content"]


def test_correction_count(collector):
    collector.record_correction("tabs", "spaces")
    collector.record_correction("tabs", "spaces")
    collector.record_correction("semicolons", "no semicolons")
    assert collector.get_correction_count() == 3
    assert collector.get_correction_count("tabs") == 2
