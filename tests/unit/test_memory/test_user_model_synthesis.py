"""Tests for enhanced user model synthesis."""
from unittest.mock import MagicMock

from homie_core.memory.user_model import UserModelSynthesizer, UserProfile, _categorize_fact


def test_user_profile_new_fields():
    profile = UserProfile()
    assert profile.expertise_level == "unknown"
    assert profile.communication_style == "balanced"
    assert profile.last_updated == ""
    assert profile.update_count == 0


def test_user_profile_context_block_includes_new_fields():
    profile = UserProfile(
        name="Alice",
        expertise_level="advanced",
        communication_style="technical",
    )
    block = profile.to_context_block()
    assert "Expertise: advanced" in block
    assert "Style: technical" in block


def test_is_empty_with_skills():
    profile = UserProfile(skills=["Python"])
    assert not profile.is_empty()


def test_incremental_update():
    mock_sm = MagicMock()
    mock_sm.get_facts.return_value = []

    synthesizer = UserModelSynthesizer(semantic_memory=mock_sm)
    new_facts = [
        "User knows Python very well",
        "User uses VS Code as their editor",
        "User prefers dark themes",
    ]
    profile = synthesizer.incremental_update(new_facts)
    assert "Python very well" in profile.skills or len(profile.all_facts) == 3
    assert profile.update_count == 1


def test_incremental_update_persists():
    mock_sm = MagicMock()
    mock_sm.get_facts.return_value = []

    synthesizer = UserModelSynthesizer(semantic_memory=mock_sm)
    synthesizer.incremental_update(["User knows Python"])
    mock_sm.set_profile.assert_called_once()
    call_args = mock_sm.set_profile.call_args
    assert call_args[0][0] == "user_model"


def test_load_persisted_profile():
    mock_sm = MagicMock()
    mock_sm.get_profile.return_value = {
        "name": "Bob",
        "role": "developer",
        "location": "NYC",
        "skills": ["Python", "Go"],
        "expertise_level": "advanced",
        "communication_style": "technical",
        "update_count": 5,
    }

    synthesizer = UserModelSynthesizer(semantic_memory=mock_sm)
    profile = synthesizer.load_persisted_profile()
    assert profile is not None
    assert profile.name == "Bob"
    assert profile.expertise_level == "advanced"
    assert profile.update_count == 5


def test_load_persisted_profile_missing():
    mock_sm = MagicMock()
    mock_sm.get_profile.return_value = None

    synthesizer = UserModelSynthesizer(semantic_memory=mock_sm)
    assert synthesizer.load_persisted_profile() is None


def test_expertise_inference():
    mock_sm = MagicMock()
    mock_sm.get_facts.return_value = [
        {"fact": f"User knows {tech}", "confidence": 0.8}
        for tech in ["Python", "Golang", "Rust", "TypeScript", "SQL",
                      "Docker", "Kubernetes", "Terraform", "Redis"]
    ]

    synthesizer = UserModelSynthesizer(semantic_memory=mock_sm)
    profile = synthesizer.get_profile()
    assert profile.expertise_level == "expert"


def test_communication_style_inference():
    mock_sm = MagicMock()
    mock_sm.get_facts.return_value = [
        {"fact": "User is a software engineer and developer", "confidence": 0.8},
        {"fact": "User works with technical code review", "confidence": 0.7},
    ]

    synthesizer = UserModelSynthesizer(semantic_memory=mock_sm)
    profile = synthesizer.get_profile()
    assert profile.communication_style == "technical"


def test_categorize_fact_patterns():
    assert _categorize_fact("User's name is Alice")[0] == "name"
    assert _categorize_fact("User is a software engineer")[0] == "role"
    assert _categorize_fact("User lives in Paris")[0] == "location"
    assert _categorize_fact("User knows Python well")[0] == "skill"
    assert _categorize_fact("User uses VS Code for editing")[0] == "tech"
    assert _categorize_fact("User prefers dark mode themes")[0] == "preference"
    assert _categorize_fact("Some random text about weather")[0] == "other"
