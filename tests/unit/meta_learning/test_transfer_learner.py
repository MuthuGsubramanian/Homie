# tests/unit/meta_learning/test_transfer_learner.py
"""Tests for the TransferLearner."""

import pytest

from homie_core.meta_learning.transfer_learner import TransferLearner, _text_overlap


class TestTransferLearner:
    def _make_learner(self, knowledge_query=None):
        return TransferLearner(knowledge_query=knowledge_query)

    def test_record_and_find_by_tag(self):
        tl = self._make_learner()
        tl.record_solution(
            domain="web",
            task_description="build caching layer",
            solution={"approach": "lru_cache"},
            tags=["caching", "performance"],
        )
        results = tl.find_analogies("improve caching speed", source_domain="web", target_domain="api")
        assert len(results) >= 1
        assert results[0]["source_domain"] == "web"

    def test_find_analogies_empty_when_no_match(self):
        tl = self._make_learner()
        tl.record_solution(
            domain="web",
            task_description="build caching layer",
            solution={"approach": "lru_cache"},
            tags=["caching"],
        )
        results = tl.find_analogies("quantum physics simulation", source_domain="science", target_domain="api")
        assert results == []

    def test_transfer_solution_marks_metadata(self):
        tl = self._make_learner()
        original = {"approach": "lru_cache", "max_size": 1000, "ttl": 60}
        adapted = tl.transfer_solution(original, from_domain="web", to_domain="api")
        assert adapted["_transferred"] is True
        assert adapted["_original_domain"] == "web"
        assert adapted["_target_domain"] == "api"
        assert "max_size" in adapted["_needs_tuning"]
        assert "ttl" in adapted["_needs_tuning"]

    def test_transfer_solution_does_not_mutate_original(self):
        tl = self._make_learner()
        original = {"approach": "retry", "count": 3}
        adapted = tl.transfer_solution(original, "a", "b")
        assert "_transferred" not in original
        assert "_transferred" in adapted

    def test_cross_domain_insights_requires_two_domains(self):
        tl = self._make_learner()
        tl.record_solution("web", "task1", {}, tags=["caching"], success_score=0.9)
        tl.record_solution("api", "task2", {}, tags=["caching"], success_score=0.8)
        tl.record_solution("web", "task3", {}, tags=["unique_tag"], success_score=0.7)

        insights = tl.get_cross_domain_insights()
        assert len(insights) == 1
        assert insights[0]["pattern"] == "caching"
        assert set(insights[0]["domains"]) == {"web", "api"}

    def test_cross_domain_insights_empty_for_single_domain(self):
        tl = self._make_learner()
        tl.record_solution("web", "task1", {}, tags=["only_web"])
        assert tl.get_cross_domain_insights() == []

    def test_knowledge_query_integration(self):
        extra = [{"source_domain": "ext", "target_domain": "t", "relevance": 0.99, "source_task": "ext_task"}]
        tl = self._make_learner(knowledge_query=lambda **kw: extra)
        tl.record_solution("ext", "ext_task", {}, tags=["match"])
        results = tl.find_analogies("match something", source_domain="ext", target_domain="t")
        # Should include both local matches and the external ones
        assert any(r.get("source_domain") == "ext" for r in results)

    def test_text_overlap_helper(self):
        assert _text_overlap("hello world", "hello world") == 1.0
        assert _text_overlap("hello world", "goodbye moon") == 0.0
        assert _text_overlap("", "anything") == 0.0
