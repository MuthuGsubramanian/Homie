# tests/unit/adaptive_learning/test_prompt_builder.py
import pytest
from homie_core.adaptive_learning.preference.prompt_builder import build_preference_prompt
from homie_core.adaptive_learning.preference.profile import PreferenceProfile


class TestBuildPreferencePrompt:
    def test_generates_prompt_for_terse_profile(self):
        profile = PreferenceProfile(verbosity=0.1, confidence=0.8)
        prompt = build_preference_prompt(profile)
        assert "concise" in prompt.lower() or "brief" in prompt.lower() or "short" in prompt.lower()

    def test_generates_prompt_for_verbose_profile(self):
        profile = PreferenceProfile(verbosity=0.9, confidence=0.8)
        prompt = build_preference_prompt(profile)
        assert "detailed" in prompt.lower() or "thorough" in prompt.lower()

    def test_includes_format_preference(self):
        profile = PreferenceProfile(format_preference="bullets", confidence=0.8)
        prompt = build_preference_prompt(profile)
        assert "bullet" in prompt.lower()

    def test_low_confidence_returns_empty(self):
        profile = PreferenceProfile(confidence=0.05)
        prompt = build_preference_prompt(profile, min_confidence=0.1)
        assert prompt == ""

    def test_returns_string(self):
        profile = PreferenceProfile(confidence=0.5)
        prompt = build_preference_prompt(profile)
        assert isinstance(prompt, str)
