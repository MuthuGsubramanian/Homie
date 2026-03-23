import pytest
from homie_core.model_evolution.milestone_tracker import MilestoneTracker


class TestMilestoneTracker:
    def test_no_milestone_initially(self):
        tracker = MilestoneTracker(min_facts=50, min_prefs=10, min_customs=3)
        assert tracker.should_rebuild() is False

    def test_facts_milestone(self):
        tracker = MilestoneTracker(min_facts=5, min_prefs=100, min_customs=100)
        for _ in range(5):
            tracker.record_new_fact()
        assert tracker.should_rebuild() is True

    def test_preference_milestone(self):
        tracker = MilestoneTracker(min_facts=100, min_prefs=3, min_customs=100)
        for _ in range(3):
            tracker.record_preference_change()
        assert tracker.should_rebuild() is True

    def test_customization_milestone(self):
        tracker = MilestoneTracker(min_facts=100, min_prefs=100, min_customs=2)
        tracker.record_new_customization()
        tracker.record_new_customization()
        assert tracker.should_rebuild() is True

    def test_reset_after_rebuild(self):
        tracker = MilestoneTracker(min_facts=2, min_prefs=100, min_customs=100)
        tracker.record_new_fact()
        tracker.record_new_fact()
        assert tracker.should_rebuild() is True
        tracker.reset()
        assert tracker.should_rebuild() is False

    def test_manual_trigger(self):
        tracker = MilestoneTracker(min_facts=100, min_prefs=100, min_customs=100)
        tracker.trigger_manual()
        assert tracker.should_rebuild() is True

    def test_get_summary(self):
        tracker = MilestoneTracker(min_facts=50, min_prefs=10, min_customs=3)
        tracker.record_new_fact()
        tracker.record_new_fact()
        summary = tracker.get_summary()
        assert summary["new_facts"] == 2
        assert summary["thresholds"]["min_facts"] == 50
