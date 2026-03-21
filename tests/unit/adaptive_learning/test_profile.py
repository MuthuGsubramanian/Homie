# tests/unit/adaptive_learning/test_profile.py
import pytest
from homie_core.adaptive_learning.preference.profile import (
    PreferenceProfile,
    PreferenceLayer,
    PreferenceResolver,
)


class TestPreferenceProfile:
    def test_defaults(self):
        p = PreferenceProfile()
        assert p.verbosity == 0.5
        assert p.formality == 0.5
        assert p.technical_depth == 0.7
        assert p.format_preference == "mixed"
        assert p.explanation_style == "top_down"
        assert p.confidence == 0.0
        assert p.sample_count == 0

    def test_to_dict_and_from_dict(self):
        p = PreferenceProfile(verbosity=0.3, formality=0.8)
        d = p.to_dict()
        p2 = PreferenceProfile.from_dict(d)
        assert p2.verbosity == 0.3
        assert p2.formality == 0.8

    def test_update_dimension(self):
        p = PreferenceProfile()
        p.update("verbosity", 0.2, learning_rate=0.3)
        assert p.verbosity != 0.5  # should have moved toward 0.2
        assert p.sample_count == 1

    def test_update_clamps_to_range(self):
        p = PreferenceProfile(verbosity=0.1)
        p.update("verbosity", -0.5, learning_rate=1.0)
        assert p.verbosity >= 0.0


class TestPreferenceResolver:
    def test_global_fallback(self):
        resolver = PreferenceResolver()
        resolver.set_profile(PreferenceLayer.GLOBAL, "default", PreferenceProfile(verbosity=0.3))
        resolved = resolver.resolve(domain=None, project=None, hour=None)
        assert resolved.verbosity == 0.3

    def test_domain_overrides_global(self):
        resolver = PreferenceResolver()
        resolver.set_profile(PreferenceLayer.GLOBAL, "default", PreferenceProfile(verbosity=0.5))
        resolver.set_profile(PreferenceLayer.DOMAIN, "coding", PreferenceProfile(verbosity=0.2))
        resolved = resolver.resolve(domain="coding", project=None, hour=None)
        assert resolved.verbosity == 0.2

    def test_project_overrides_domain(self):
        resolver = PreferenceResolver()
        resolver.set_profile(PreferenceLayer.GLOBAL, "default", PreferenceProfile(verbosity=0.5))
        resolver.set_profile(PreferenceLayer.DOMAIN, "coding", PreferenceProfile(verbosity=0.3))
        resolver.set_profile(PreferenceLayer.PROJECT, "homie", PreferenceProfile(verbosity=0.1))
        resolved = resolver.resolve(domain="coding", project="homie", hour=None)
        assert resolved.verbosity == 0.1

    def test_missing_layer_falls_through(self):
        resolver = PreferenceResolver()
        resolver.set_profile(PreferenceLayer.GLOBAL, "default", PreferenceProfile(verbosity=0.5))
        resolved = resolver.resolve(domain="unknown", project=None, hour=None)
        assert resolved.verbosity == 0.5  # falls to global
