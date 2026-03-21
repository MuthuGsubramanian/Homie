"""Tests for priority_inference module."""
from __future__ import annotations

import pytest
from homie_core.intelligence.priority_inference import PriorityInference, PriorityItem


class TestPriorityInference:
    def _infer(self, **kwargs) -> list[PriorityItem]:
        return PriorityInference().infer(**kwargs)

    # --- commitment scoring ---

    def test_commitment_with_today_scores_above_0_9(self):
        items = self._infer(commitments=[{"text": "send the report", "due_by": "today"}])
        assert len(items) >= 1
        assert items[0].score > 0.9

    def test_commitment_with_now_scores_above_0_9(self):
        items = self._infer(commitments=[{"text": "call Alice", "due_by": "now"}])
        assert items[0].score > 0.9

    def test_commitment_with_asap_scores_above_0_9(self):
        items = self._infer(commitments=[{"text": "fix production bug", "due_by": "asap"}])
        assert items[0].score > 0.9

    def test_commitment_with_urgent_scores_above_0_9(self):
        items = self._infer(commitments=[{"text": "deploy hotfix", "due_by": "urgent"}])
        assert items[0].score > 0.9

    def test_commitment_with_tomorrow_scores_above_0_8(self):
        items = self._infer(commitments=[{"text": "review PR", "due_by": "tomorrow"}])
        assert items[0].score > 0.8

    def test_commitment_without_due_by_scores_0_8(self):
        items = self._infer(commitments=[{"text": "write docs"}])
        assert items[0].score == pytest.approx(0.8)

    def test_commitment_reason_is_commitment(self):
        items = self._infer(commitments=[{"text": "finish report", "due_by": "today"}])
        assert items[0].reason == "commitment"

    def test_commitment_source_passed_through(self):
        items = self._infer(commitments=[{"text": "review", "source": "github"}])
        assert items[0].source == "github"

    # --- stuck task scoring ---

    def test_stuck_task_scores_0_9(self):
        items = self._infer(incomplete_tasks=[{"description": "fix auth bug", "state": "stuck"}])
        assert items[0].score == pytest.approx(0.9)

    def test_stuck_task_reason_is_stuck(self):
        items = self._infer(incomplete_tasks=[{"description": "fix auth bug", "state": "stuck"}])
        assert items[0].reason == "stuck"

    def test_active_task_scores_0_6(self):
        items = self._infer(incomplete_tasks=[{"description": "write tests", "state": "active"}])
        assert items[0].score == pytest.approx(0.6)

    def test_active_task_reason_is_incomplete(self):
        items = self._infer(incomplete_tasks=[{"description": "write tests", "state": "active"}])
        assert items[0].reason == "incomplete"

    def test_task_description_fallback_to_task_key(self):
        items = self._infer(incomplete_tasks=[{"task": "deploy service"}])
        assert items[0].description == "deploy service"

    # --- recurrent topic scoring ---

    def test_recurrent_topics_have_decreasing_scores(self):
        items = self._infer(recent_topics=["topic A", "topic B", "topic C"])
        recurrent = [i for i in items if i.reason == "recurrent"]
        scores = [i.score for i in recurrent]
        assert scores == sorted(scores, reverse=True)

    def test_first_recurrent_topic_scores_0_7(self):
        items = self._infer(recent_topics=["machine learning"])
        assert items[0].score == pytest.approx(0.7)

    def test_recurrent_topic_reason(self):
        items = self._infer(recent_topics=["deployment"])
        assert items[0].reason == "recurrent"

    def test_recurrent_score_floor_is_0_3(self):
        # 10 topics: score = max(0.3, 0.7 - i*0.1), floor kicks in at i>=4
        items = self._infer(recent_topics=[f"topic {i}" for i in range(10)])
        recurrent = [i for i in items if i.reason == "recurrent"]
        assert all(i.score >= 0.3 for i in recurrent)

    # --- sorting ---

    def test_results_sorted_by_score_descending(self):
        items = self._infer(
            commitments=[{"text": "finish docs", "due_by": "today"}],
            incomplete_tasks=[{"description": "write tests", "state": "active"}],
            recent_topics=["refactoring"],
        )
        scores = [i.score for i in items]
        assert scores == sorted(scores, reverse=True)

    def test_empty_inputs_return_empty_list(self):
        items = self._infer()
        assert items == []

    def test_none_inputs_return_empty_list(self):
        items = self._infer(commitments=None, incomplete_tasks=None, recent_topics=None)
        assert items == []

    def test_mixed_none_and_data(self):
        items = self._infer(
            commitments=None,
            incomplete_tasks=[{"description": "deploy", "state": "stuck"}],
            recent_topics=None,
        )
        assert len(items) == 1
        assert items[0].score == pytest.approx(0.9)

    # --- PriorityItem dataclass ---

    def test_priority_item_default_source(self):
        p = PriorityItem(description="foo", score=0.5, reason="recurrent")
        assert p.source == ""
