"""Tests for SocialIntelligence analysis engine."""
import json
from unittest.mock import MagicMock
from homie_core.social_media.intelligence import SocialIntelligence, _extract_topics, SocialProfile
from homie_core.social_media.models import SocialPost, ProfileInfo, ProfileStats
from homie_core.social_media.provider import FeedProvider, ProfileProvider


class TestExtractTopics:
    def test_extracts_hashtags(self):
        topics = _extract_topics(["Love #python and #coding", "More #python stuff"])
        assert "python" in topics
        assert "coding" in topics

    def test_extracts_keywords(self):
        topics = _extract_topics(["Machine learning is amazing technology"])
        assert "machine" in topics or "learning" in topics or "amazing" in topics

    def test_filters_stop_words(self):
        topics = _extract_topics(["this that with from have been just about"])
        assert topics == []

    def test_empty_input(self):
        assert _extract_topics([]) == []


class TestSocialIntelligence:
    def test_analyze_profiles(self):
        vault = MagicMock()
        vault.get_credential.return_value = None
        intel = SocialIntelligence(vault)

        # Create a mock that passes isinstance checks for both ProfileProvider and FeedProvider
        class _DummyProvider(ProfileProvider, FeedProvider):
            def get_profile(self, username=None): ...
            def get_stats(self): ...
            def get_feed(self, limit=20): ...
            def search_posts(self, query, limit=10): ...

        provider = MagicMock(spec=_DummyProvider)
        provider.get_profile.return_value = ProfileInfo(
            platform="twitter", username="@test", display_name="Test", bio="I love python",
        )
        provider.get_stats.return_value = ProfileStats(
            platform="twitter", followers=1000, post_count=200,
        )
        provider.get_feed.return_value = [
            SocialPost(id="1", platform="twitter", author="@test",
                       content="Python is great #python", timestamp=1700000000.0),
            SocialPost(id="2", platform="twitter", author="@test",
                       content="Working on AI stuff #machinelearning", timestamp=1700001000.0),
        ]

        result = intel.analyze_profiles({"twitter": provider})

        assert "twitter" in result.platforms
        assert result.platforms["twitter"].audience_size == 1000
        assert len(result.cross_platform.primary_topics) > 0
        # Verify stored to vault
        vault.store_credential.assert_called_once()

    def test_analyze_multiple_platforms(self):
        vault = MagicMock()
        intel = SocialIntelligence(vault)

        class _DummyProvider(ProfileProvider, FeedProvider):
            def get_profile(self, username=None): ...
            def get_stats(self): ...
            def get_feed(self, limit=20): ...
            def search_posts(self, query, limit=10): ...

        twitter = MagicMock(spec=_DummyProvider)
        twitter.get_stats.return_value = ProfileStats(platform="twitter", followers=500)
        twitter.get_feed.return_value = [
            SocialPost(id="1", platform="twitter", author="t", content="#tech news", timestamp=0),
        ]

        reddit = MagicMock(spec=_DummyProvider)
        reddit.get_stats.return_value = ProfileStats(platform="reddit", followers=200)
        reddit.get_feed.return_value = [
            SocialPost(id="2", platform="reddit", author="r", content="#tech discussion", timestamp=0),
        ]

        result = intel.analyze_profiles({"twitter": twitter, "reddit": reddit})
        assert "twitter" in result.platforms
        assert "reddit" in result.platforms
        assert "tech" in result.cross_platform.primary_topics

    def test_handles_provider_errors(self):
        vault = MagicMock()
        intel = SocialIntelligence(vault)

        provider = MagicMock()
        provider.get_stats.side_effect = RuntimeError("API down")
        provider.get_feed.side_effect = RuntimeError("API down")

        result = intel.analyze_profiles({"broken": provider})
        assert "broken" in result.platforms
        assert result.platforms["broken"].audience_size == 0


class TestGetCachedProfile:
    def test_returns_none_when_no_cache(self):
        vault = MagicMock()
        vault.get_credential.return_value = None
        intel = SocialIntelligence(vault)
        assert intel.get_cached_profile() is None

    def test_returns_cached_profile(self):
        vault = MagicMock()
        cred = MagicMock()
        cred.access_token = json.dumps({
            "last_scan": "2024-01-01T00:00:00Z",
            "platforms": {
                "twitter": {"topics": ["python", "tech"], "tone": "casual",
                            "audience_size": 1000, "engagement_rate": 0.05},
            },
            "cross_platform": {"primary_topics": ["python", "tech"]},
        })
        vault.get_credential.return_value = cred
        intel = SocialIntelligence(vault)
        result = intel.get_cached_profile()

        assert result is not None
        assert "twitter" in result.platforms
        assert result.platforms["twitter"].audience_size == 1000
        assert "python" in result.cross_platform.primary_topics

    def test_handles_corrupt_cache(self):
        vault = MagicMock()
        cred = MagicMock()
        cred.access_token = "not valid json{{"
        vault.get_credential.return_value = cred
        intel = SocialIntelligence(vault)
        assert intel.get_cached_profile() is None
