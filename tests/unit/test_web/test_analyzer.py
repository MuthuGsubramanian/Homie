"""Tests for WebAnalyzer and WebPageAnalysis."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from homie_core.web.analyzer import WebAnalyzer
from homie_core.web.models import WebPageAnalysis

_SAMPLE_HTML = """<!DOCTYPE html>
<html><head>
<title>Test Page</title>
<meta name="description" content="A test page">
<meta property="og:title" content="OG Test">
<meta property="og:type" content="article">
</head><body>
<nav>Skip nav</nav>
<article>
<h1>Main Heading</h1>
<h2>Sub Heading</h2>
<p>This is the main article content with enough text.</p>
<a href="/link1">Link 1</a>
<a href="/link2">Link 2</a>
<img src="/img1.jpg">
</article>
<footer>Skip footer</footer>
</body></html>"""


class TestWebPageAnalysis:
    def test_to_dict(self) -> None:
        analysis = WebPageAnalysis(
            url="https://example.com",
            title="Example",
            page_type="webpage",
            description="desc",
            main_content="content",
            headings=["h1"],
            links_count=3,
            images_count=1,
            og_data={"og:title": "Ex"},
            analyzed_at=1000.0,
        )
        d = analysis.to_dict()
        assert d["url"] == "https://example.com"
        assert d["title"] == "Example"
        assert d["page_type"] == "webpage"
        assert d["description"] == "desc"
        assert d["main_content"] == "content"
        assert d["headings"] == ["h1"]
        assert d["links_count"] == 3
        assert d["images_count"] == 1
        assert d["og_data"] == {"og:title": "Ex"}
        assert d["analyzed_at"] == 1000.0


class TestWebAnalyzer:
    def _mock_response(self, text: str, content_type: str = "text/html") -> MagicMock:
        resp = MagicMock()
        resp.text = text
        resp.headers = {"Content-Type": content_type}
        resp.raise_for_status = MagicMock()
        return resp

    @patch("homie_core.web.analyzer.requests.get")
    def test_analyze_url_success(self, mock_get: MagicMock) -> None:
        mock_get.return_value = self._mock_response(_SAMPLE_HTML)
        analyzer = WebAnalyzer()
        result = analyzer.analyze_url("https://example.com/article")

        assert result.title == "Test Page"
        assert result.description == "A test page"
        assert result.page_type == "article"
        assert "Main Heading" in result.headings
        assert "Sub Heading" in result.headings
        assert result.links_count == 2
        assert result.images_count == 1
        assert result.og_data == {"og:title": "OG Test", "og:type": "article"}
        assert "main article content" in result.main_content
        # Skipped tags content should NOT be in main_content
        assert "Skip nav" not in result.main_content
        assert "Skip footer" not in result.main_content

    @patch("homie_core.web.analyzer.requests.get")
    def test_analyze_url_non_html(self, mock_get: MagicMock) -> None:
        mock_get.return_value = self._mock_response('{"key": "value"}', "application/json")
        analyzer = WebAnalyzer()
        result = analyzer.analyze_url("https://api.example.com/data")

        assert result.page_type == "non-html"
        assert "application/json" in result.description

    @patch("homie_core.web.analyzer.requests.get")
    def test_analyze_url_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = ConnectionError("Network failure")
        analyzer = WebAnalyzer()
        result = analyzer.analyze_url("https://unreachable.test")

        assert result.page_type == "error"
        assert "Network failure" in result.description
