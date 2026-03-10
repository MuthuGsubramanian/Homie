from unittest.mock import MagicMock

from homie_core.intelligence.proactive_retrieval import ProactiveRetrieval


def test_stage_context_on_change():
    sm = MagicMock()
    sm.get_facts.return_value = [{"fact": "User prefers dark mode", "confidence": 0.8}]
    em = MagicMock()
    em.recall.return_value = [{"summary": "Worked on engine.py yesterday"}]

    pr = ProactiveRetrieval(semantic_memory=sm, episodic_memory=em)
    pr.on_context_change(process="Code.exe", title="engine.py - Homie")
    staged = pr.get_staged_context()

    assert len(staged["facts"]) == 1
    assert len(staged["episodes"]) == 1


def test_staged_context_is_cleared_on_consume():
    sm = MagicMock()
    sm.get_facts.return_value = [{"fact": "test"}]
    em = MagicMock()
    em.recall.return_value = []

    pr = ProactiveRetrieval(semantic_memory=sm, episodic_memory=em)
    pr.on_context_change(process="Code.exe", title="engine.py")
    pr.consume_staged_context()
    staged = pr.get_staged_context()
    assert staged["facts"] == []
    assert staged["episodes"] == []


def test_no_query_when_same_context():
    sm = MagicMock()
    sm.get_facts.return_value = []
    em = MagicMock()
    em.recall.return_value = []

    pr = ProactiveRetrieval(semantic_memory=sm, episodic_memory=em)
    pr.on_context_change(process="Code.exe", title="engine.py")
    pr.on_context_change(process="Code.exe", title="engine.py")
    assert sm.get_facts.call_count == 1


def test_builds_query_from_title():
    sm = MagicMock()
    sm.get_facts.return_value = []
    em = MagicMock()
    em.recall.return_value = []

    pr = ProactiveRetrieval(semantic_memory=sm, episodic_memory=em)
    pr.on_context_change(process="Code.exe", title="config.py - Homie")
    em.recall.assert_called_once()
    query_arg = em.recall.call_args[0][0]
    assert "config.py" in query_arg or "Homie" in query_arg
