from unittest.mock import MagicMock
from homie_core.memory.consolidator import MemoryConsolidator
from homie_core.memory.working import WorkingMemory


def test_create_session_digest():
    working = WorkingMemory()
    working.add_message("user", "How do I fix this auth bug?")
    working.add_message("assistant", "Check the token validation logic")
    working.add_message("user", "That fixed it, thanks!")
    working.update("active_app", "VS Code")
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "User debugged an authentication token validation issue in VS Code and resolved it."
    consolidator = MemoryConsolidator(model_engine=mock_engine)
    digest = consolidator.create_session_digest(working)
    assert digest is not None
    assert "summary" in digest
    mock_engine.generate.assert_called_once()


def test_extract_facts_from_digest():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = '["User knows Python authentication patterns", "User uses VS Code"]'
    consolidator = MemoryConsolidator(model_engine=mock_engine)
    facts = consolidator.extract_facts("User debugged auth in VS Code using Python")
    assert isinstance(facts, list)
    assert len(facts) == 2


def test_extract_facts_handles_invalid_json():
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "not valid json"
    consolidator = MemoryConsolidator(model_engine=mock_engine)
    facts = consolidator.extract_facts("some summary")
    assert facts == []


def test_empty_conversation_returns_empty_digest():
    working = WorkingMemory()
    mock_engine = MagicMock()
    consolidator = MemoryConsolidator(model_engine=mock_engine)
    digest = consolidator.create_session_digest(working)
    assert digest["summary"] == ""
    mock_engine.generate.assert_not_called()
