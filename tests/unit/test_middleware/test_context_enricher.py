"""Tests for ContextEnricherMiddleware."""
from __future__ import annotations

import pytest

from homie_core.knowledge.graph import KnowledgeGraph
from homie_core.knowledge.models import Entity
from homie_core.middleware.context_enricher import ContextEnricherMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(tmp_path) -> KnowledgeGraph:
    return KnowledgeGraph(tmp_path / "kg.db")


BASE_PROMPT = "You are Homie."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContextEnricherNoSources:
    """With no sources, prompt must be returned unchanged."""

    def test_no_sources_prompt_unchanged(self):
        mw = ContextEnricherMiddleware()
        result = mw.modify_prompt(BASE_PROMPT)
        assert result == BASE_PROMPT

    def test_empty_string_prompt_unchanged(self):
        mw = ContextEnricherMiddleware()
        assert mw.modify_prompt("") == ""


class TestContextEnricherWithGraph:
    def test_graph_no_entities_prompt_unchanged(self, tmp_path):
        graph = _make_graph(tmp_path)
        mw = ContextEnricherMiddleware(graph=graph)
        result = mw.modify_prompt(BASE_PROMPT)
        assert result == BASE_PROMPT

    def test_graph_with_entities_injects_recent_topics(self, tmp_path):
        graph = _make_graph(tmp_path)
        graph.merge_entity(Entity(name="Alice", entity_type="person"))
        graph.merge_entity(Entity(name="Homie", entity_type="project"))

        mw = ContextEnricherMiddleware(graph=graph)
        result = mw.modify_prompt(BASE_PROMPT)

        assert "[LIVE CONTEXT]" in result
        assert "Recent topics:" in result
        assert "Alice" in result
        assert "Homie" in result

    def test_graph_entities_include_type(self, tmp_path):
        graph = _make_graph(tmp_path)
        graph.merge_entity(Entity(name="Budget", entity_type="concept"))

        mw = ContextEnricherMiddleware(graph=graph)
        result = mw.modify_prompt(BASE_PROMPT)
        assert "(concept)" in result

    def test_graph_limits_to_five_entities(self, tmp_path):
        graph = _make_graph(tmp_path)
        for i in range(10):
            graph.merge_entity(Entity(name=f"Entity{i}", entity_type="concept"))

        mw = ContextEnricherMiddleware(graph=graph)
        result = mw.modify_prompt(BASE_PROMPT)
        # Count occurrences of "- Entity" lines — should be at most 5
        lines = [ln for ln in result.splitlines() if ln.strip().startswith("- ")]
        assert len(lines) <= 5


class TestContextEnricherWithEmailSummary:
    def test_email_summary_injected(self):
        mw = ContextEnricherMiddleware(email_summary_fn=lambda: "3 unread from boss")
        result = mw.modify_prompt(BASE_PROMPT)
        assert "Email: 3 unread from boss" in result
        assert "[LIVE CONTEXT]" in result

    def test_email_summary_none_skipped(self):
        mw = ContextEnricherMiddleware(email_summary_fn=lambda: None)
        result = mw.modify_prompt(BASE_PROMPT)
        assert result == BASE_PROMPT

    def test_email_summary_exception_silently_skipped(self):
        def boom():
            raise RuntimeError("network failure")

        mw = ContextEnricherMiddleware(email_summary_fn=boom)
        result = mw.modify_prompt(BASE_PROMPT)
        assert result == BASE_PROMPT

    def test_email_empty_string_skipped(self):
        mw = ContextEnricherMiddleware(email_summary_fn=lambda: "")
        result = mw.modify_prompt(BASE_PROMPT)
        assert result == BASE_PROMPT


class TestContextEnricherWithBehavioral:
    def test_behavioral_summary_injected(self):
        mw = ContextEnricherMiddleware(behavioral_summary_fn=lambda: "coding for 2h")
        result = mw.modify_prompt(BASE_PROMPT)
        assert "Activity: coding for 2h" in result

    def test_behavioral_summary_exception_silently_skipped(self):
        def boom():
            raise ValueError("permission denied")

        mw = ContextEnricherMiddleware(behavioral_summary_fn=boom)
        result = mw.modify_prompt(BASE_PROMPT)
        assert result == BASE_PROMPT


class TestContextEnricherMultipleSources:
    def test_all_sources_combined(self, tmp_path):
        graph = _make_graph(tmp_path)
        graph.merge_entity(Entity(name="PR#42", entity_type="task"))

        mw = ContextEnricherMiddleware(
            graph=graph,
            email_summary_fn=lambda: "inbox: 5 new",
            behavioral_summary_fn=lambda: "flow state",
        )
        result = mw.modify_prompt(BASE_PROMPT)

        assert "[LIVE CONTEXT]" in result
        assert "Recent topics:" in result
        assert "Email: inbox: 5 new" in result
        assert "Activity: flow state" in result

    def test_context_appended_after_prompt(self):
        mw = ContextEnricherMiddleware(email_summary_fn=lambda: "1 new email")
        result = mw.modify_prompt(BASE_PROMPT)
        assert result.startswith(BASE_PROMPT)

    def test_middleware_attributes(self):
        mw = ContextEnricherMiddleware()
        assert mw.name == "context_enricher"
        assert mw.order == 25
