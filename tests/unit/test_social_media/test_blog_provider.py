"""Tests for Blog provider (RSS/Atom)."""
from unittest.mock import patch, MagicMock
from homie_core.social_media.blog_provider import BlogProvider
from homie_core.social_media.provider import FeedProvider, ProfileProvider, PublishProvider, DirectMessageProvider

_SAMPLE_FEED = {
    "feed": {"title": "My Blog", "subtitle": "Tech thoughts", "author": "Test Author",
             "link": "https://blog.example.com"},
    "entries": [
        {"id": "1", "title": "Post 1", "summary": "Content about python",
         "link": "https://blog.example.com/1", "author": "Test Author",
         "published_parsed": (2024, 1, 1, 0, 0, 0, 0, 1, 0)},
        {"id": "2", "title": "Post 2", "summary": "Content about AI",
         "link": "https://blog.example.com/2", "author": "Test Author",
         "published_parsed": (2024, 1, 2, 0, 0, 0, 0, 2, 0)},
    ],
}


class TestBlogConnect:
    @patch("homie_core.social_media.blog_provider.feedparser")
    def test_connect_success(self, mock_fp):
        mock_fp.parse.return_value = _SAMPLE_FEED
        cred = MagicMock()
        cred.access_token = "https://blog.example.com/feed.xml"
        p = BlogProvider()
        assert p.connect(cred) is True
        assert p.is_connected is True

    @patch("homie_core.social_media.blog_provider.feedparser")
    def test_connect_empty_feed(self, mock_fp):
        mock_fp.parse.return_value = {"feed": {}, "entries": []}
        cred = MagicMock()
        cred.access_token = "https://bad.example.com/feed.xml"
        p = BlogProvider()
        assert p.connect(cred) is False


class TestBlogFeed:
    @patch("homie_core.social_media.blog_provider.feedparser")
    def test_get_feed(self, mock_fp):
        mock_fp.parse.return_value = _SAMPLE_FEED
        p = BlogProvider()
        p._feed_url = "https://blog.example.com/feed.xml"
        p._connected = True
        posts = p.get_feed()
        assert len(posts) == 2
        assert posts[0].content == "Content about python"
        assert posts[0].post_type == "article"
        assert posts[0].platform == "blog"
        assert posts[0].url == "https://blog.example.com/1"

    @patch("homie_core.social_media.blog_provider.feedparser")
    def test_get_feed_with_limit(self, mock_fp):
        mock_fp.parse.return_value = _SAMPLE_FEED
        p = BlogProvider()
        p._feed_url = "https://blog.example.com/feed.xml"
        p._connected = True
        posts = p.get_feed(limit=1)
        assert len(posts) == 1


class TestBlogSearch:
    @patch("homie_core.social_media.blog_provider.feedparser")
    def test_search_posts(self, mock_fp):
        mock_fp.parse.return_value = _SAMPLE_FEED
        p = BlogProvider()
        p._feed_url = "https://blog.example.com/feed.xml"
        p._connected = True
        results = p.search_posts("python")
        assert len(results) == 1
        assert "python" in results[0].content.lower()


class TestBlogProfile:
    @patch("homie_core.social_media.blog_provider.feedparser")
    def test_get_profile(self, mock_fp):
        mock_fp.parse.return_value = _SAMPLE_FEED
        p = BlogProvider()
        p._feed_url = "https://blog.example.com/feed.xml"
        p._connected = True
        info = p.get_profile()
        assert info.display_name == "My Blog"
        assert info.bio == "Tech thoughts"
        assert info.platform == "blog"

    @patch("homie_core.social_media.blog_provider.feedparser")
    def test_get_stats(self, mock_fp):
        mock_fp.parse.return_value = _SAMPLE_FEED
        p = BlogProvider()
        p._feed_url = "https://blog.example.com/feed.xml"
        p._connected = True
        stats = p.get_stats()
        assert stats.post_count == 2
        assert stats.platform == "blog"


class TestBlogCapabilities:
    def test_is_feed_and_profile_only(self):
        p = BlogProvider()
        assert isinstance(p, FeedProvider)
        assert isinstance(p, ProfileProvider)
        assert not isinstance(p, PublishProvider)
        assert not isinstance(p, DirectMessageProvider)
