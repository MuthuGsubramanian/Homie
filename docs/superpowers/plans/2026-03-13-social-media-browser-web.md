# Social Media Profiles, URL Analysis & Browser History — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 social media platform integrations, a URL analysis engine, and a browser history module to Homie.

**Architecture:** Three new modules (`social_media`, `web`, `browser`) following existing patterns — capability-based provider ABCs, service facades with vault storage, AI tools via ToolRegistry, daemon sync callbacks, CLI subparsers.

**Tech Stack:** Python 3.12, requests, feedparser (new dep), sqlite3, html.parser (stdlib), dataclasses

**Spec:** `docs/superpowers/specs/2026-03-13-social-media-profiles-design.md`

---

## Chunk 1: Foundation — Models, ABCs, and Web Analyzer

### Task 1: Social Media Data Models

**Files:**
- Create: `src/homie_core/social_media/__init__.py` (empty placeholder)
- Create: `src/homie_core/social_media/models.py`
- Create: `tests/unit/test_social_media/__init__.py`
- Create: `tests/unit/test_social_media/test_models.py`

- [ ] **Step 1: Write failing test for SocialPost**

```python
# tests/unit/test_social_media/test_models.py
"""Tests for social media data models."""
from homie_core.social_media.models import SocialPost

class TestSocialPost:
    def test_to_dict(self):
        post = SocialPost(
            id="p1", platform="twitter", author="@user",
            content="Hello world", timestamp=1700000000.0,
        )
        d = post.to_dict()
        assert d["id"] == "p1"
        assert d["platform"] == "twitter"
        assert d["author"] == "@user"
        assert d["content"] == "Hello world"
        assert d["likes"] == 0
        assert d["post_type"] == "text"

    def test_to_dict_with_optionals(self):
        post = SocialPost(
            id="p2", platform="reddit", author="u/test",
            content="Check this", timestamp=1700000000.0,
            url="https://reddit.com/r/test/1", likes=42, shares=5,
            media_urls=["https://i.imgur.com/a.jpg"], post_type="image",
        )
        d = post.to_dict()
        assert d["url"] == "https://reddit.com/r/test/1"
        assert d["likes"] == 42
        assert d["media_urls"] == ["https://i.imgur.com/a.jpg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_social_media/test_models.py::TestSocialPost -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement SocialPost**

```python
# src/homie_core/social_media/__init__.py
"""Social media profile integrations — provider abstraction, intelligence, and tools."""

# src/homie_core/social_media/models.py
"""Data models for social media integrations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class SocialPost:
    id: str
    platform: str
    author: str
    content: str
    timestamp: float
    url: str | None = None
    media_urls: list[str] = field(default_factory=list)
    likes: int = 0
    comments: int = 0
    shares: int = 0
    post_type: str = "text"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "platform": self.platform, "author": self.author,
            "content": self.content, "timestamp": self.timestamp,
            "url": self.url, "media_urls": self.media_urls,
            "likes": self.likes, "comments": self.comments,
            "shares": self.shares, "post_type": self.post_type,
        }
```

- [ ] **Step 4: Run test — should pass**

Run: `python -m pytest tests/unit/test_social_media/test_models.py::TestSocialPost -v`

- [ ] **Step 5: Add tests for ProfileInfo, ProfileStats, Notification, Conversation, DirectMessage**

```python
# append to tests/unit/test_social_media/test_models.py
from homie_core.social_media.models import (
    ProfileInfo, ProfileStats, Notification, Conversation, DirectMessage,
)

class TestProfileInfo:
    def test_to_dict(self):
        info = ProfileInfo(
            platform="twitter", username="@test", display_name="Test User",
            bio="Hello", verified=True,
        )
        d = info.to_dict()
        assert d["username"] == "@test"
        assert d["verified"] is True
        assert d["avatar_url"] is None

class TestProfileStats:
    def test_to_dict(self):
        stats = ProfileStats(platform="twitter", followers=1000, post_count=50)
        d = stats.to_dict()
        assert d["followers"] == 1000
        assert d["engagement_rate"] == 0.0

class TestNotification:
    def test_to_dict(self):
        n = Notification(
            id="n1", platform="twitter", type="mention",
            sender="@other", content="tagged you", timestamp=1700000000.0,
        )
        d = n.to_dict()
        assert d["type"] == "mention"

class TestConversation:
    def test_to_dict(self):
        c = Conversation(
            id="c1", platform="twitter", participants=["@a", "@b"],
            last_message_preview="Hey!", last_activity=1700000000.0,
        )
        d = c.to_dict()
        assert d["participants"] == ["@a", "@b"]

class TestDirectMessage:
    def test_to_dict(self):
        dm = DirectMessage(
            id="dm1", platform="twitter", conversation_id="c1",
            sender="@a", content="Hello", timestamp=1700000000.0,
        )
        d = dm.to_dict()
        assert d["conversation_id"] == "c1"
```

- [ ] **Step 6: Implement remaining models**

```python
# append to src/homie_core/social_media/models.py

@dataclass
class ProfileInfo:
    platform: str
    username: str
    display_name: str
    bio: str
    avatar_url: str | None = None
    profile_url: str | None = None
    joined: float | None = None
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform, "username": self.username,
            "display_name": self.display_name, "bio": self.bio,
            "avatar_url": self.avatar_url, "profile_url": self.profile_url,
            "joined": self.joined, "verified": self.verified,
        }

@dataclass
class ProfileStats:
    platform: str
    followers: int = 0
    following: int = 0
    post_count: int = 0
    engagement_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform, "followers": self.followers,
            "following": self.following, "post_count": self.post_count,
            "engagement_rate": self.engagement_rate,
        }

@dataclass
class Notification:
    id: str
    platform: str
    type: str
    sender: str
    content: str
    timestamp: float
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "platform": self.platform, "type": self.type,
            "sender": self.sender, "content": self.content,
            "timestamp": self.timestamp, "url": self.url,
        }

@dataclass
class Conversation:
    id: str
    platform: str
    participants: list[str]
    last_message_preview: str
    last_activity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "platform": self.platform,
            "participants": self.participants,
            "last_message_preview": self.last_message_preview,
            "last_activity": self.last_activity,
        }

@dataclass
class DirectMessage:
    id: str
    platform: str
    conversation_id: str
    sender: str
    content: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "platform": self.platform,
            "conversation_id": self.conversation_id,
            "sender": self.sender, "content": self.content,
            "timestamp": self.timestamp,
        }
```

- [ ] **Step 7: Run all model tests**

Run: `python -m pytest tests/unit/test_social_media/test_models.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/homie_core/social_media/__init__.py src/homie_core/social_media/models.py tests/unit/test_social_media/__init__.py tests/unit/test_social_media/test_models.py
git commit -m "feat(social-media): add data models — SocialPost, ProfileInfo, ProfileStats, Notification, Conversation, DirectMessage"
```

---

### Task 2: Capability Interface ABCs

**Files:**
- Create: `src/homie_core/social_media/provider.py`
- Test: `tests/unit/test_social_media/test_provider.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_social_media/test_provider.py
"""Tests for social media provider ABCs."""
from homie_core.social_media.provider import (
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
    PublishProvider, DirectMessageProvider,
)
from homie_core.social_media.models import SocialPost, ProfileInfo, ProfileStats, Conversation, DirectMessage

class ConcreteAll(SocialMediaProviderBase, FeedProvider, ProfileProvider, PublishProvider, DirectMessageProvider):
    platform_name = "test"
    def get_feed(self, limit=20): return []
    def search_posts(self, query, limit=10): return []
    def get_profile(self, username=None): return ProfileInfo(platform="test", username="u", display_name="U", bio="")
    def get_stats(self): return ProfileStats(platform="test")
    def publish(self, content, media_urls=None): return {"status": "ok"}
    def list_conversations(self, limit=20): return []
    def get_messages(self, conversation_id, limit=20): return []
    def send_message(self, recipient, text): return {"status": "sent"}

class TestProviderBase:
    def test_connect_and_properties(self):
        p = ConcreteAll()
        assert p.is_connected is False
        from unittest.mock import MagicMock
        cred = MagicMock()
        cred.access_token = "tok123"
        result = p.connect(cred)
        assert result is True
        assert p.is_connected is True
        assert p._token == "tok123"

    def test_capability_check(self):
        p = ConcreteAll()
        assert isinstance(p, FeedProvider)
        assert isinstance(p, ProfileProvider)
        assert isinstance(p, PublishProvider)
        assert isinstance(p, DirectMessageProvider)

class FeedOnly(SocialMediaProviderBase, FeedProvider):
    platform_name = "feedonly"
    def get_feed(self, limit=20): return []
    def search_posts(self, query, limit=10): return []

class TestPartialCapabilities:
    def test_feed_only_is_not_publish(self):
        p = FeedOnly()
        assert isinstance(p, FeedProvider)
        assert not isinstance(p, PublishProvider)
```

- [ ] **Step 2: Run test — should fail**

Run: `python -m pytest tests/unit/test_social_media/test_provider.py -v`

- [ ] **Step 3: Implement provider ABCs**

```python
# src/homie_core/social_media/provider.py
"""Abstract base classes for social media providers."""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from homie_core.social_media.models import (
    SocialPost, ProfileInfo, ProfileStats, Conversation, DirectMessage,
)

logger = logging.getLogger(__name__)

class SocialMediaProviderBase:
    """Base class all providers inherit — handles credential lifecycle."""
    platform_name: str = ""

    def __init__(self):
        self._token: str | None = None
        self._connected: bool = False

    def connect(self, credential) -> bool:
        try:
            self._token = credential.access_token
            self._connected = True
            return True
        except Exception:
            logger.exception("Failed to connect %s", self.platform_name)
            return False

    def refresh_token(self) -> bool:
        return False  # Override in providers that support refresh

    @property
    def is_connected(self) -> bool:
        return self._connected

class FeedProvider(ABC):
    @abstractmethod
    def get_feed(self, limit: int = 20) -> list[SocialPost]: ...
    @abstractmethod
    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]: ...

class ProfileProvider(ABC):
    @abstractmethod
    def get_profile(self, username: str | None = None) -> ProfileInfo: ...
    @abstractmethod
    def get_stats(self) -> ProfileStats: ...

class PublishProvider(ABC):
    @abstractmethod
    def publish(self, content: str, media_urls: list[str] | None = None) -> dict: ...

class DirectMessageProvider(ABC):
    @abstractmethod
    def list_conversations(self, limit: int = 20) -> list[Conversation]: ...
    @abstractmethod
    def get_messages(self, conversation_id: str, limit: int = 20) -> list[DirectMessage]: ...
    @abstractmethod
    def send_message(self, recipient: str, text: str) -> dict: ...
```

- [ ] **Step 4: Run tests — should pass**

Run: `python -m pytest tests/unit/test_social_media/test_provider.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/social_media/provider.py tests/unit/test_social_media/test_provider.py
git commit -m "feat(social-media): add capability ABCs — FeedProvider, ProfileProvider, PublishProvider, DirectMessageProvider"
```

---

### Task 3: Web Analyzer — Models and Engine

**Files:**
- Create: `src/homie_core/web/__init__.py`
- Create: `src/homie_core/web/models.py`
- Create: `src/homie_core/web/analyzer.py`
- Create: `tests/unit/test_web/__init__.py`
- Create: `tests/unit/test_web/test_analyzer.py`

- [ ] **Step 1: Write failing test for WebPageAnalysis model**

```python
# tests/unit/test_web/test_analyzer.py
"""Tests for WebAnalyzer."""
from homie_core.web.models import WebPageAnalysis

class TestWebPageAnalysis:
    def test_to_dict(self):
        a = WebPageAnalysis(
            url="https://example.com", title="Example", page_type="article",
            description="An example page", main_content="Hello world",
            headings=["Heading 1"], links_count=5, images_count=2,
            og_data={"og:title": "Example"}, analyzed_at=1700000000.0,
        )
        d = a.to_dict()
        assert d["url"] == "https://example.com"
        assert d["page_type"] == "article"
        assert d["headings"] == ["Heading 1"]
```

- [ ] **Step 2: Run test — should fail**

Run: `python -m pytest tests/unit/test_web/test_analyzer.py::TestWebPageAnalysis -v`

- [ ] **Step 3: Implement WebPageAnalysis model**

```python
# src/homie_core/web/__init__.py
"""Web analysis utilities."""

# src/homie_core/web/models.py
"""Data models for web analysis."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class WebPageAnalysis:
    url: str
    title: str
    page_type: str
    description: str
    main_content: str
    headings: list[str]
    links_count: int
    images_count: int
    og_data: dict
    analyzed_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url, "title": self.title, "page_type": self.page_type,
            "description": self.description, "main_content": self.main_content,
            "headings": self.headings, "links_count": self.links_count,
            "images_count": self.images_count, "og_data": self.og_data,
            "analyzed_at": self.analyzed_at,
        }
```

- [ ] **Step 4: Run test — should pass**

- [ ] **Step 5: Write failing test for WebAnalyzer**

```python
# append to tests/unit/test_web/test_analyzer.py
from unittest.mock import patch, MagicMock
from homie_core.web.analyzer import WebAnalyzer

_SAMPLE_HTML = """<!DOCTYPE html>
<html><head>
<title>Test Page</title>
<meta name="description" content="A test page">
<meta property="og:title" content="OG Test">
<meta property="og:type" content="article">
</head><body>
<nav>Skip this nav</nav>
<article>
<h1>Main Heading</h1>
<h2>Sub Heading</h2>
<p>This is the main article content with enough text to be meaningful.</p>
<a href="/link1">Link 1</a>
<a href="/link2">Link 2</a>
<img src="/img1.jpg">
</article>
<footer>Skip footer</footer>
</body></html>"""

class TestWebAnalyzer:
    @patch("homie_core.web.analyzer.requests")
    def test_analyze_url_success(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = _SAMPLE_HTML
        resp.headers = {"content-type": "text/html"}
        mock_requests.get.return_value = resp

        analyzer = WebAnalyzer()
        result = analyzer.analyze_url("https://example.com/article")

        assert result.title == "Test Page"
        assert result.description == "A test page"
        assert "Main Heading" in result.headings
        assert "Sub Heading" in result.headings
        assert result.links_count == 2
        assert result.images_count == 1
        assert result.og_data.get("og:title") == "OG Test"
        assert "main article content" in result.main_content

    @patch("homie_core.web.analyzer.requests")
    def test_analyze_url_non_html(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = '{"key": "value"}'
        resp.headers = {"content-type": "application/json"}
        mock_requests.get.return_value = resp

        analyzer = WebAnalyzer()
        result = analyzer.analyze_url("https://api.example.com/data")
        assert result.page_type == "non-html"

    @patch("homie_core.web.analyzer.requests")
    def test_analyze_url_error(self, mock_requests):
        mock_requests.get.side_effect = Exception("Connection error")

        analyzer = WebAnalyzer()
        result = analyzer.analyze_url("https://bad.example.com")
        assert result.page_type == "error"
        assert "Connection error" in result.description
```

- [ ] **Step 6: Implement WebAnalyzer**

```python
# src/homie_core/web/analyzer.py
"""Web page analysis engine."""
from __future__ import annotations
import logging
import time
from html.parser import HTMLParser
from typing import Any

import requests

from homie_core.web.models import WebPageAnalysis

logger = logging.getLogger(__name__)
_MAX_CONTENT = 5000


class _ContentExtractor(HTMLParser):
    """Extract structured content from HTML."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self.description = ""
        self.og_data: dict[str, str] = {}
        self.headings: list[str] = []
        self.links_count = 0
        self.images_count = 0
        self._text_parts: list[str] = []
        self._current_tag = ""
        self._in_title = False
        self._skip_tags = {"nav", "footer", "header", "aside", "script", "style"}
        self._skip_depth = 0
        self._in_article = False
        self._article_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self._current_tag = tag
        attr_dict = dict(attrs)

        if tag in self._skip_tags:
            self._skip_depth += 1
            return

        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = attr_dict.get("name", "")
            prop = attr_dict.get("property", "")
            content = attr_dict.get("content", "")
            if name == "description":
                self.description = content or ""
            if prop and prop.startswith("og:"):
                self.og_data[prop] = content or ""
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            pass  # text captured in handle_data
        elif tag == "a":
            self.links_count += 1
        elif tag == "img":
            self.images_count += 1
        elif tag in ("article", "main"):
            self._in_article = True

    def handle_endtag(self, tag: str):
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in ("article", "main"):
            self._in_article = False
        self._current_tag = ""

    def handle_data(self, data: str):
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += text
            return
        if self._skip_depth > 0:
            return
        if self._current_tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.headings.append(text)
        if self._in_article:
            self._article_text.append(text)
        self._text_parts.append(text)

    @property
    def main_content(self) -> str:
        source = self._article_text if self._article_text else self._text_parts
        content = " ".join(source)
        return content[:_MAX_CONTENT]


def _classify_page(og_data: dict, url: str, content: str) -> str:
    og_type = og_data.get("og:type", "")
    if og_type:
        return og_type
    url_lower = url.lower()
    if "/docs" in url_lower or "/documentation" in url_lower:
        return "documentation"
    if "/product" in url_lower or "/shop" in url_lower:
        return "product"
    if any(x in url_lower for x in ("/video", "youtube.com", "vimeo.com")):
        return "video"
    return "webpage"


class WebAnalyzer:
    """Fetch and analyze web pages."""

    def analyze_url(self, url: str) -> WebPageAnalysis:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Homie/1.0"})
            content_type = resp.headers.get("content-type", "")
            if "html" not in content_type:
                return WebPageAnalysis(
                    url=url, title="", page_type="non-html",
                    description=f"Content-Type: {content_type}",
                    main_content=resp.text[:_MAX_CONTENT],
                    headings=[], links_count=0, images_count=0,
                    og_data={}, analyzed_at=time.time(),
                )
            extractor = _ContentExtractor()
            extractor.feed(resp.text)
            page_type = _classify_page(extractor.og_data, url, extractor.main_content)
            return WebPageAnalysis(
                url=url, title=extractor.title, page_type=page_type,
                description=extractor.description,
                main_content=extractor.main_content,
                headings=extractor.headings,
                links_count=extractor.links_count,
                images_count=extractor.images_count,
                og_data=extractor.og_data,
                analyzed_at=time.time(),
            )
        except Exception as exc:
            logger.exception("Failed to analyze URL %s", url)
            return WebPageAnalysis(
                url=url, title="", page_type="error",
                description=str(exc), main_content="",
                headings=[], links_count=0, images_count=0,
                og_data={}, analyzed_at=time.time(),
            )
```

- [ ] **Step 7: Run tests — should pass**

Run: `python -m pytest tests/unit/test_web/ -v`

- [ ] **Step 8: Commit**

```bash
git add src/homie_core/web/ tests/unit/test_web/
git commit -m "feat(web): add WebAnalyzer engine with HTML content extraction and classification"
```

---

### Task 4: Web Analyzer Tools

**Files:**
- Create: `src/homie_core/web/tools.py`
- Create: `tests/unit/test_web/test_tools.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_web/test_tools.py
"""Tests for web analysis AI tools."""
import json
from unittest.mock import MagicMock
from homie_core.brain.tool_registry import ToolRegistry
from homie_core.web.tools import register_web_tools
from homie_core.web.models import WebPageAnalysis

class TestWebToolRegistration:
    def test_registers_web_analyze_tool(self):
        registry = ToolRegistry()
        analyzer = MagicMock()
        register_web_tools(registry, analyzer)
        names = {t.name for t in registry.list_tools()}
        assert "web_analyze" in names

class TestWebAnalyzeTool:
    def test_returns_json(self):
        registry = ToolRegistry()
        analyzer = MagicMock()
        analyzer.analyze_url.return_value = WebPageAnalysis(
            url="https://example.com", title="Example", page_type="webpage",
            description="Test", main_content="Content here",
            headings=["H1"], links_count=3, images_count=1,
            og_data={}, analyzed_at=1700000000.0,
        )
        register_web_tools(registry, analyzer)
        tool = registry.get("web_analyze")
        result = tool.execute(url="https://example.com")
        data = json.loads(result)
        assert data["title"] == "Example"
        assert data["page_type"] == "webpage"
```

- [ ] **Step 2: Run test — should fail**

- [ ] **Step 3: Implement**

```python
# src/homie_core/web/tools.py
"""AI tool wrappers for web analysis."""
from __future__ import annotations
import json
from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry

_MAX_OUTPUT = 2000

def _truncate(text: str) -> str:
    return text[:_MAX_OUTPUT] + "..." if len(text) > _MAX_OUTPUT else text

def register_web_tools(registry: ToolRegistry, web_analyzer) -> None:
    def tool_web_analyze(url: str) -> str:
        result = web_analyzer.analyze_url(url)
        return _truncate(json.dumps(result.to_dict()))

    registry.register(Tool(
        name="web_analyze",
        description="Fetch and analyze a webpage — extracts title, content, headings, links, images, and metadata.",
        params=[ToolParam(name="url", description="URL to analyze", type="string")],
        execute=tool_web_analyze,
        category="web",
    ))
```

- [ ] **Step 4: Run tests — should pass**

Run: `python -m pytest tests/unit/test_web/test_tools.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/web/tools.py tests/unit/test_web/test_tools.py
git commit -m "feat(web): add web_analyze AI tool"
```

---

## Chunk 2: Social Media OAuth, Providers, and Service

### Task 5: Shared OAuth Helper

**Files:**
- Create: `src/homie_core/social_media/oauth.py`
- Test: `tests/unit/test_social_media/test_oauth.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_social_media/test_oauth.py
"""Tests for social media OAuth helpers."""
from unittest.mock import patch, MagicMock
from homie_core.social_media.oauth import SocialMediaOAuth

class TestSocialMediaOAuth:
    def test_get_auth_url(self):
        oauth = SocialMediaOAuth(
            platform="twitter",
            client_id="cid",
            client_secret="csec",
            auth_url="https://twitter.com/i/oauth2/authorize",
            token_url="https://api.twitter.com/2/oauth2/token",
            scopes=["tweet.read", "users.read"],
            redirect_port=8551,
        )
        url = oauth.get_auth_url()
        assert "client_id=cid" in url
        assert "tweet.read" in url
        assert "8551" in url

    @patch("homie_core.social_media.oauth.requests")
    def test_exchange(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"access_token": "tok", "refresh_token": "ref"}
        mock_requests.post.return_value = resp

        oauth = SocialMediaOAuth(
            platform="twitter", client_id="cid", client_secret="csec",
            auth_url="https://x.com/auth", token_url="https://x.com/token",
            scopes=["tweet.read"], redirect_port=8551,
        )
        tokens = oauth.exchange("code123")
        assert tokens["access_token"] == "tok"

    @patch("homie_core.social_media.oauth.requests")
    def test_refresh(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"access_token": "new_tok", "refresh_token": "new_ref"}
        mock_requests.post.return_value = resp

        oauth = SocialMediaOAuth(
            platform="reddit", client_id="cid", client_secret="csec",
            auth_url="https://reddit.com/auth", token_url="https://reddit.com/token",
            scopes=["read"], redirect_port=8552,
        )
        tokens = oauth.refresh("old_ref")
        assert tokens["access_token"] == "new_tok"
```

- [ ] **Step 2: Run test — should fail**

- [ ] **Step 3: Implement**

```python
# src/homie_core/social_media/oauth.py
"""Shared OAuth helpers for social media platforms."""
from __future__ import annotations
import logging
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import requests as _requests

logger = logging.getLogger(__name__)


class _CallbackHandler(BaseHTTPRequestHandler):
    code: str | None = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        _CallbackHandler.code = params.get("code", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>Success! You can close this tab.</h1>")

    def log_message(self, format, *args):
        pass  # Suppress server logs


class SocialMediaOAuth:
    def __init__(self, platform: str, client_id: str, client_secret: str,
                 auth_url: str, token_url: str, scopes: list[str],
                 redirect_port: int):
        self.platform = platform
        self._client_id = client_id
        self._client_secret = client_secret
        self._auth_url = auth_url
        self._token_url = token_url
        self._scopes = scopes
        self._redirect_port = redirect_port
        self._redirect_uri = f"http://localhost:{redirect_port}/callback"

    def get_auth_url(self) -> str:
        params = {
            "client_id": self._client_id,
            "response_type": "code",
            "redirect_uri": self._redirect_uri,
            "scope": " ".join(self._scopes),
        }
        return f"{self._auth_url}?{urllib.parse.urlencode(params)}"

    def wait_for_redirect(self, timeout: int = 120) -> str | None:
        _CallbackHandler.code = None
        server = HTTPServer(("127.0.0.1", self._redirect_port), _CallbackHandler)
        server.timeout = timeout
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        server.server_close()
        return _CallbackHandler.code

    def exchange(self, code: str) -> dict[str, Any]:
        resp = _requests.post(self._token_url, data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._redirect_uri,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        resp = _requests.post(self._token_url, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()
```

- [ ] **Step 4: Run tests — should pass**

Run: `python -m pytest tests/unit/test_social_media/test_oauth.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/social_media/oauth.py tests/unit/test_social_media/test_oauth.py
git commit -m "feat(social-media): add shared SocialMediaOAuth helper with token exchange and refresh"
```

---

### Task 6: Twitter Provider

**Files:**
- Create: `src/homie_core/social_media/twitter_provider.py`
- Test: `tests/unit/test_social_media/test_twitter_provider.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_social_media/test_twitter_provider.py
"""Tests for Twitter/X provider."""
from unittest.mock import patch, MagicMock
from homie_core.social_media.twitter_provider import TwitterProvider

def _mock_cred():
    cred = MagicMock()
    cred.access_token = "bearer_test"
    cred.refresh_token = "refresh_test"
    return cred

class TestTwitterConnect:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_connect_success(self, mock_req):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": {"id": "123", "username": "testuser", "name": "Test"}}
        mock_req.get.return_value = resp

        p = TwitterProvider()
        assert p.connect(_mock_cred()) is True
        assert p._user_id == "123"
        assert p._username == "testuser"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_connect_failure(self, mock_req):
        mock_req.get.side_effect = Exception("Network error")
        p = TwitterProvider()
        assert p.connect(_mock_cred()) is False

class TestTwitterFeed:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_feed(self, mock_req):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": [
            {"id": "t1", "text": "Hello Twitter", "created_at": "2024-01-01T00:00:00Z",
             "author_id": "123", "public_metrics": {"like_count": 5, "reply_count": 1, "retweet_count": 2}},
        ]}
        mock_req.get.return_value = resp

        p = TwitterProvider()
        p._token = "tok"
        p._connected = True
        p._user_id = "123"
        posts = p.get_feed(limit=5)
        assert len(posts) == 1
        assert posts[0].content == "Hello Twitter"
        assert posts[0].likes == 5

class TestTwitterProfile:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_get_profile(self, mock_req):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": {
            "id": "123", "username": "testuser", "name": "Test User",
            "description": "Bio here", "verified": True,
            "profile_image_url": "https://pbs.twimg.com/img.jpg",
            "public_metrics": {"followers_count": 1000, "following_count": 500, "tweet_count": 300},
        }}
        mock_req.get.return_value = resp

        p = TwitterProvider()
        p._token = "tok"
        p._connected = True
        info = p.get_profile()
        assert info.username == "testuser"
        assert info.verified is True

class TestTwitterPublish:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_publish(self, mock_req):
        resp = MagicMock()
        resp.status_code = 201
        resp.json.return_value = {"data": {"id": "new_tweet_id"}}
        mock_req.post.return_value = resp

        p = TwitterProvider()
        p._token = "tok"
        p._connected = True
        result = p.publish("Hello from Homie!")
        assert result["status"] == "published"
        assert result["post_id"] == "new_tweet_id"

class TestTwitterDM:
    @patch("homie_core.social_media.twitter_provider.requests")
    def test_send_message(self, mock_req):
        resp = MagicMock()
        resp.status_code = 201
        resp.json.return_value = {"data": {"dm_event_id": "dm123"}}
        mock_req.post.return_value = resp

        p = TwitterProvider()
        p._token = "tok"
        p._connected = True
        result = p.send_message("456", "Hello!")
        assert result["status"] == "sent"

    @patch("homie_core.social_media.twitter_provider.requests")
    def test_rate_limit_retry(self, mock_req):
        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "1"}
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"data": []}
        mock_req.get.side_effect = [rate_resp, ok_resp]

        p = TwitterProvider()
        p._token = "tok"
        p._connected = True
        p._user_id = "123"
        with patch("homie_core.social_media.twitter_provider.time.sleep"):
            posts = p.get_feed()
        assert posts == []
```

- [ ] **Step 2: Run test — should fail**

- [ ] **Step 3: Implement TwitterProvider**

```python
# src/homie_core/social_media/twitter_provider.py
"""Twitter/X provider — Feed, Profile, Publish, DM."""
from __future__ import annotations
import logging
import time
from typing import Any

import requests

from homie_core.social_media.models import (
    SocialPost, ProfileInfo, ProfileStats, Conversation, DirectMessage,
)
from homie_core.social_media.provider import (
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
    PublishProvider, DirectMessageProvider,
)

logger = logging.getLogger(__name__)
_API = "https://api.twitter.com/2"


class TwitterProvider(
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
    PublishProvider, DirectMessageProvider,
):
    platform_name = "twitter"

    def __init__(self):
        super().__init__()
        self._user_id: str | None = None
        self._username: str | None = None
        self._refresh_token_str: str | None = None

    def connect(self, credential) -> bool:
        try:
            self._token = credential.access_token
            self._refresh_token_str = getattr(credential, "refresh_token", None)
            data = self._call("GET", "/users/me",
                              params={"user.fields": "id,username,name"})
            self._user_id = data["data"]["id"]
            self._username = data["data"]["username"]
            self._connected = True
            return True
        except Exception:
            logger.exception("Twitter connect failed")
            self._connected = False
            return False

    def _call(self, method: str, path: str, params: dict | None = None,
              json_body: dict | None = None, retries: int = 2) -> dict:
        url = f"{_API}{path}"
        headers = {"Authorization": f"Bearer {self._token}"}
        for attempt in range(retries + 1):
            if method == "GET":
                resp = requests.get(url, headers=headers, params=params, timeout=15)
            else:
                resp = requests.post(url, headers=headers, json=json_body, timeout=15)
            if resp.status_code == 429 and attempt < retries:
                wait = int(resp.headers.get("Retry-After", 5))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        return {}

    # --- FeedProvider ---
    def get_feed(self, limit: int = 20) -> list[SocialPost]:
        try:
            data = self._call("GET", f"/users/{self._user_id}/timelines/reverse_chronological",
                              params={"max_results": min(limit, 100),
                                      "tweet.fields": "created_at,author_id,public_metrics"})
            posts = []
            for t in data.get("data", []):
                metrics = t.get("public_metrics", {})
                posts.append(SocialPost(
                    id=t["id"], platform="twitter", author=t.get("author_id", ""),
                    content=t.get("text", ""), timestamp=0,
                    likes=metrics.get("like_count", 0),
                    comments=metrics.get("reply_count", 0),
                    shares=metrics.get("retweet_count", 0),
                ))
            return posts
        except Exception:
            logger.exception("Twitter get_feed failed")
            return []

    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]:
        try:
            data = self._call("GET", "/tweets/search/recent",
                              params={"query": query, "max_results": min(limit, 100),
                                      "tweet.fields": "created_at,author_id,public_metrics"})
            return [SocialPost(id=t["id"], platform="twitter", author=t.get("author_id", ""),
                               content=t.get("text", ""), timestamp=0,
                               likes=t.get("public_metrics", {}).get("like_count", 0))
                    for t in data.get("data", [])]
        except Exception:
            logger.exception("Twitter search failed")
            return []

    # --- ProfileProvider ---
    def get_profile(self, username: str | None = None) -> ProfileInfo:
        target = username or self._username or "me"
        path = f"/users/by/username/{target}" if username else "/users/me"
        data = self._call("GET", path, params={
            "user.fields": "id,username,name,description,verified,profile_image_url,public_metrics",
        })
        u = data["data"]
        return ProfileInfo(
            platform="twitter", username=u.get("username", ""),
            display_name=u.get("name", ""), bio=u.get("description", ""),
            avatar_url=u.get("profile_image_url"),
            profile_url=f"https://x.com/{u.get('username', '')}",
            verified=u.get("verified", False),
        )

    def get_stats(self) -> ProfileStats:
        data = self._call("GET", "/users/me",
                          params={"user.fields": "public_metrics"})
        m = data["data"].get("public_metrics", {})
        return ProfileStats(
            platform="twitter", followers=m.get("followers_count", 0),
            following=m.get("following_count", 0), post_count=m.get("tweet_count", 0),
        )

    # --- PublishProvider ---
    def publish(self, content: str, media_urls: list[str] | None = None) -> dict:
        try:
            data = self._call("POST", "/tweets", json_body={"text": content})
            tweet_id = data.get("data", {}).get("id", "")
            return {"status": "published", "platform": "twitter", "post_id": tweet_id,
                    "url": f"https://x.com/i/status/{tweet_id}"}
        except Exception as exc:
            return {"status": "error", "platform": "twitter", "error": str(exc)}

    # --- DirectMessageProvider ---
    def list_conversations(self, limit: int = 20) -> list[Conversation]:
        try:
            data = self._call("GET", f"/dm_conversations",
                              params={"dm_event.fields": "created_at,sender_id,text"})
            convos = []
            for c in data.get("data", [])[:limit]:
                convos.append(Conversation(
                    id=c.get("id", ""), platform="twitter",
                    participants=c.get("participant_ids", []),
                    last_message_preview=c.get("text", "")[:100],
                    last_activity=0,
                ))
            return convos
        except Exception:
            logger.exception("Twitter list_conversations failed")
            return []

    def get_messages(self, conversation_id: str, limit: int = 20) -> list[DirectMessage]:
        try:
            data = self._call("GET", f"/dm_conversations/{conversation_id}/dm_events",
                              params={"max_results": min(limit, 100)})
            return [DirectMessage(
                id=e.get("id", ""), platform="twitter",
                conversation_id=conversation_id,
                sender=e.get("sender_id", ""), content=e.get("text", ""),
                timestamp=0,
            ) for e in data.get("data", [])]
        except Exception:
            logger.exception("Twitter get_messages failed")
            return []

    def send_message(self, recipient: str, text: str) -> dict:
        try:
            data = self._call("POST", "/dm_conversations/with/{}/messages".format(recipient),
                              json_body={"text": text})
            msg_id = data.get("data", {}).get("dm_event_id", "")
            return {"status": "sent", "platform": "twitter", "message_id": msg_id}
        except Exception as exc:
            return {"status": "error", "platform": "twitter", "error": str(exc)}
```

- [ ] **Step 4: Run tests — should pass**

Run: `python -m pytest tests/unit/test_social_media/test_twitter_provider.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/social_media/twitter_provider.py tests/unit/test_social_media/test_twitter_provider.py
git commit -m "feat(social-media): add Twitter/X provider — feed, profile, publish, DM"
```

---

### Task 7: Reddit Provider

**Files:**
- Create: `src/homie_core/social_media/reddit_provider.py`
- Test: `tests/unit/test_social_media/test_reddit_provider.py`

Same structure as Twitter — implements all 4 capabilities. Key differences:
- API base: `https://oauth.reddit.com`
- Auth header: `Bearer` token
- User-Agent required: `Homie/1.0 by <username>`
- Feed = subreddit posts from subscriptions
- Publish = submit to subreddit
- DM = Reddit messaging API

- [ ] **Step 1: Write failing tests** (same pattern as Twitter — test connect, feed, profile, publish, DM, rate limit)

- [ ] **Step 2: Implement RedditProvider** — follows exact same _call() pattern with 429 retry

- [ ] **Step 3: Run tests — should pass**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(social-media): add Reddit provider — feed, profile, publish, DM"
```

---

### Task 8: LinkedIn Provider

**Files:**
- Create: `src/homie_core/social_media/linkedin_provider.py`
- Test: `tests/unit/test_social_media/test_linkedin_provider.py`

Implements Feed, Profile, Publish (no DM). Key differences:
- API base: `https://api.linkedin.com/v2`
- Profile uses `/me` endpoint
- Publish uses UGC Posts API
- No DM capability (API restriction)

- [ ] **Step 1-4:** Same TDD cycle as Twitter/Reddit

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(social-media): add LinkedIn provider — feed, profile, publish (no DM)"
```

---

### Task 9: Facebook Provider

**Files:**
- Create: `src/homie_core/social_media/facebook_provider.py`
- Test: `tests/unit/test_social_media/test_facebook_provider.py`

All 4 capabilities via Meta Graph API:
- API base: `https://graph.facebook.com/v19.0`
- Feed = `/me/feed`
- Profile = `/me?fields=id,name,email`
- Publish = `/me/feed` POST
- DM = Messenger via `/me/conversations`

- [ ] **Step 1-4:** Same TDD cycle

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(social-media): add Facebook provider — feed, profile, publish, DM via Graph API"
```

---

### Task 10: Instagram Provider

**Files:**
- Create: `src/homie_core/social_media/instagram_provider.py`
- Test: `tests/unit/test_social_media/test_instagram_provider.py`

All 4 capabilities via Instagram Graph API (shared Meta token):
- API base: `https://graph.facebook.com/v19.0` (Instagram Graph API)
- Feed = `/me/media`
- Profile = `/me?fields=id,username,media_count`
- Publish/DM = business accounts only (returns error for personal accounts)

- [ ] **Step 1-4:** Same TDD cycle, including test for business-account-required error

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(social-media): add Instagram provider — feed, profile, publish*, DM* (*business accounts)"
```

---

### Task 11: Blog Provider

**Files:**
- Create: `src/homie_core/social_media/blog_provider.py`
- Test: `tests/unit/test_social_media/test_blog_provider.py`

Feed + Profile only (read-only, no OAuth):
- Uses `feedparser` to parse RSS/Atom
- `connect()` takes credential with `access_token` = feed URL
- `get_profile()` returns feed metadata (title, description, author)
- `get_stats()` returns post count and update frequency

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_social_media/test_blog_provider.py
from unittest.mock import patch, MagicMock
from homie_core.social_media.blog_provider import BlogProvider

_SAMPLE_FEED = {
    "feed": {"title": "My Blog", "subtitle": "Tech thoughts", "author": "Test Author"},
    "entries": [
        {"id": "1", "title": "Post 1", "summary": "Content 1",
         "link": "https://blog.example.com/1", "published_parsed": (2024, 1, 1, 0, 0, 0, 0, 0, 0)},
        {"id": "2", "title": "Post 2", "summary": "Content 2",
         "link": "https://blog.example.com/2", "published_parsed": (2024, 1, 2, 0, 0, 0, 0, 0, 0)},
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

class TestBlogFeed:
    @patch("homie_core.social_media.blog_provider.feedparser")
    def test_get_feed(self, mock_fp):
        mock_fp.parse.return_value = _SAMPLE_FEED
        p = BlogProvider()
        p._feed_url = "https://blog.example.com/feed.xml"
        p._connected = True
        posts = p.get_feed()
        assert len(posts) == 2
        assert posts[0].content == "Content 1"

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
```

- [ ] **Step 2: Implement BlogProvider**

```python
# src/homie_core/social_media/blog_provider.py
"""Blog provider — RSS/Atom feed reader (read-only)."""
from __future__ import annotations
import calendar
import logging
import time

import feedparser

from homie_core.social_media.models import SocialPost, ProfileInfo, ProfileStats
from homie_core.social_media.provider import (
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
)

logger = logging.getLogger(__name__)


class BlogProvider(SocialMediaProviderBase, FeedProvider, ProfileProvider):
    platform_name = "blog"

    def __init__(self):
        super().__init__()
        self._feed_url: str | None = None
        self._feed_data: dict | None = None

    def connect(self, credential) -> bool:
        try:
            self._feed_url = credential.access_token  # URL stored as token
            self._feed_data = feedparser.parse(self._feed_url)
            self._connected = bool(self._feed_data.get("entries"))
            return self._connected
        except Exception:
            logger.exception("Blog connect failed")
            return False

    def _fetch(self) -> dict:
        if self._feed_url:
            self._feed_data = feedparser.parse(self._feed_url)
        return self._feed_data or {}

    def get_feed(self, limit: int = 20) -> list[SocialPost]:
        data = self._fetch()
        posts = []
        for entry in data.get("entries", [])[:limit]:
            ts = 0.0
            if entry.get("published_parsed"):
                ts = calendar.timegm(entry["published_parsed"])
            posts.append(SocialPost(
                id=entry.get("id", entry.get("link", "")),
                platform="blog", author=entry.get("author", ""),
                content=entry.get("summary", entry.get("title", "")),
                timestamp=ts, url=entry.get("link"),
                post_type="article",
            ))
        return posts

    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]:
        q = query.lower()
        return [p for p in self.get_feed(limit=100)
                if q in p.content.lower() or q in (p.url or "").lower()][:limit]

    def get_profile(self, username: str | None = None) -> ProfileInfo:
        data = self._fetch()
        feed = data.get("feed", {})
        return ProfileInfo(
            platform="blog", username=self._feed_url or "",
            display_name=feed.get("title", ""),
            bio=feed.get("subtitle", feed.get("description", "")),
            profile_url=feed.get("link", self._feed_url),
        )

    def get_stats(self) -> ProfileStats:
        data = self._fetch()
        return ProfileStats(
            platform="blog", post_count=len(data.get("entries", [])),
        )
```

- [ ] **Step 3: Run tests — should pass**

- [ ] **Step 4: Commit**

```bash
git add src/homie_core/social_media/blog_provider.py tests/unit/test_social_media/test_blog_provider.py
git commit -m "feat(social-media): add Blog provider — RSS/Atom feed reader (read-only)"
```

---

### Task 12: Social Intelligence Layer

**Files:**
- Create: `src/homie_core/social_media/intelligence.py`
- Test: `tests/unit/test_social_media/test_intelligence.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_social_media/test_intelligence.py
"""Tests for SocialIntelligence analysis engine."""
from unittest.mock import MagicMock
from homie_core.social_media.intelligence import SocialIntelligence
from homie_core.social_media.models import SocialPost, ProfileInfo, ProfileStats

class TestSocialIntelligence:
    def test_analyze_profiles(self):
        vault = MagicMock()
        vault.get_credential.return_value = None
        intel = SocialIntelligence(vault)

        provider = MagicMock()
        provider.get_profile.return_value = ProfileInfo(
            platform="twitter", username="@test", display_name="Test",
            bio="I love python and AI",
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

    def test_get_cached_profile_none(self):
        vault = MagicMock()
        vault.get_credential.return_value = None
        intel = SocialIntelligence(vault)
        assert intel.get_cached_profile() is None

    def test_get_cached_profile_exists(self):
        vault = MagicMock()
        cred = MagicMock()
        cred.access_token = '{"last_scan":"2024-01-01","platforms":{},"cross_platform":{"primary_topics":["python"]}}'
        vault.get_credential.return_value = cred
        intel = SocialIntelligence(vault)
        result = intel.get_cached_profile()
        assert result is not None
```

- [ ] **Step 2: Implement SocialIntelligence**

```python
# src/homie_core/social_media/intelligence.py
"""Cross-platform social intelligence analysis."""
from __future__ import annotations
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from homie_core.social_media.provider import (
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
)

logger = logging.getLogger(__name__)


@dataclass
class PlatformProfile:
    topics: list[str] = field(default_factory=list)
    tone: str = "neutral"
    avg_posts_per_week: float = 0.0
    peak_hours: list[int] = field(default_factory=list)
    audience_size: int = 0
    engagement_rate: float = 0.0
    content_types: dict[str, float] = field(default_factory=dict)


@dataclass
class CrossPlatformAnalysis:
    primary_topics: list[str] = field(default_factory=list)
    peak_hours: list[int] = field(default_factory=list)
    platform_preferences: dict[str, str] = field(default_factory=dict)
    audience_overlap: dict = field(default_factory=dict)
    recommended_posting_times: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class SocialProfile:
    platforms: dict[str, PlatformProfile] = field(default_factory=dict)
    cross_platform: CrossPlatformAnalysis = field(default_factory=CrossPlatformAnalysis)


def _extract_topics(texts: list[str]) -> list[str]:
    words = Counter()
    for text in texts:
        hashtags = re.findall(r"#(\w+)", text.lower())
        words.update(hashtags)
        for word in re.findall(r"\b[a-z]{4,}\b", text.lower()):
            if word not in {"this", "that", "with", "from", "have", "been", "just", "about",
                            "what", "when", "they", "will", "your", "more", "some", "like"}:
                words[word] += 1
    return [w for w, _ in words.most_common(10)]


class SocialIntelligence:
    def __init__(self, vault):
        self._vault = vault

    def analyze_profiles(self, providers: dict[str, SocialMediaProviderBase]) -> SocialProfile:
        profile = SocialProfile()
        all_topics: list[str] = []

        for name, provider in providers.items():
            plat = PlatformProfile()
            if isinstance(provider, ProfileProvider):
                try:
                    stats = provider.get_stats()
                    plat.audience_size = stats.followers
                    plat.engagement_rate = stats.engagement_rate
                except Exception:
                    pass
            if isinstance(provider, FeedProvider):
                try:
                    posts = provider.get_feed(limit=50)
                    texts = [p.content for p in posts]
                    plat.topics = _extract_topics(texts)
                    all_topics.extend(plat.topics)
                    type_counts: dict[str, int] = {}
                    for p in posts:
                        type_counts[p.post_type] = type_counts.get(p.post_type, 0) + 1
                    total = len(posts) or 1
                    plat.content_types = {k: round(v / total, 2) for k, v in type_counts.items()}
                except Exception:
                    pass
            profile.platforms[name] = plat

        # Cross-platform analysis
        profile.cross_platform.primary_topics = [
            t for t, _ in Counter(all_topics).most_common(10)
        ]

        # Store result
        self._store(profile)
        return profile

    def get_cached_profile(self) -> SocialProfile | None:
        cred = self._vault.get_credential("social_intelligence", "profile")
        if not cred:
            return None
        try:
            data = json.loads(cred.access_token)
            profile = SocialProfile()
            profile.cross_platform = CrossPlatformAnalysis(
                primary_topics=data.get("cross_platform", {}).get("primary_topics", []),
            )
            for name, pdata in data.get("platforms", {}).items():
                profile.platforms[name] = PlatformProfile(
                    topics=pdata.get("topics", []),
                    tone=pdata.get("tone", "neutral"),
                    audience_size=pdata.get("audience_size", 0),
                )
            return profile
        except Exception:
            logger.exception("Failed to load cached social profile")
            return None

    def _store(self, profile: SocialProfile) -> None:
        data = {
            "last_scan": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platforms": {
                name: {"topics": p.topics, "tone": p.tone,
                       "audience_size": p.audience_size, "engagement_rate": p.engagement_rate}
                for name, p in profile.platforms.items()
            },
            "cross_platform": {
                "primary_topics": profile.cross_platform.primary_topics,
            },
        }
        try:
            self._vault.store_credential(
                provider="social_intelligence", account_id="profile",
                token_type="data", access_token=json.dumps(data),
                refresh_token="", scopes=[],
            )
        except Exception:
            logger.exception("Failed to store social intelligence profile")
```

- [ ] **Step 3: Run tests — should pass**

- [ ] **Step 4: Commit**

```bash
git add src/homie_core/social_media/intelligence.py tests/unit/test_social_media/test_intelligence.py
git commit -m "feat(social-media): add SocialIntelligence — cross-platform behavioral analysis"
```

---

### Task 13: SocialMediaService Facade

**Files:**
- Modify: `src/homie_core/social_media/__init__.py`
- Test: `tests/unit/test_social_media/test_service.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_social_media/test_service.py
"""Tests for SocialMediaService facade."""
from unittest.mock import MagicMock, patch
from homie_core.social_media import SocialMediaService
from homie_core.social_media.models import SocialPost, ProfileInfo

class TestSocialMediaServiceInit:
    @patch("homie_core.social_media.twitter_provider.TwitterProvider")
    def test_initialize_connects_twitter(self, MockTwitter):
        vault = MagicMock()
        cred = MagicMock()
        cred.active = True
        cred.account_id = "myaccount"
        vault.list_credentials.return_value = [cred]

        mock_provider = MagicMock()
        mock_provider.connect.return_value = True
        MockTwitter.return_value = mock_provider

        service = SocialMediaService(vault=vault)
        connected = service.initialize()
        assert "twitter" in connected

class TestSocialMediaServiceFeed:
    def test_get_feed(self):
        vault = MagicMock()
        provider = MagicMock()
        provider.get_feed.return_value = [
            SocialPost(id="1", platform="twitter", author="@test",
                       content="Hello", timestamp=1700000000.0),
        ]
        service = SocialMediaService(vault=vault)
        service._providers["twitter"] = provider
        feed = service.get_feed(platform="twitter")
        assert len(feed) == 1
        assert feed[0]["content"] == "Hello"

class TestSocialMediaServicePublish:
    def test_publish(self):
        vault = MagicMock()
        provider = MagicMock()
        provider.publish.return_value = {"status": "published", "post_id": "123"}
        service = SocialMediaService(vault=vault)
        service._providers["twitter"] = provider
        result = service.publish("twitter", "Hello!")
        assert result["status"] == "published"

    def test_publish_no_provider(self):
        vault = MagicMock()
        service = SocialMediaService(vault=vault)
        result = service.publish("twitter", "Hello!")
        assert result["status"] == "error"

class TestSocialMediaServiceSyncTick:
    def test_sync_tick_no_providers(self):
        vault = MagicMock()
        service = SocialMediaService(vault=vault)
        assert service.sync_tick() == "No social media platforms connected"

class TestSocialMediaServiceScan:
    def test_scan_profiles(self):
        vault = MagicMock()
        vault.get_credential.return_value = None
        provider = MagicMock()
        provider.get_profile.return_value = ProfileInfo(
            platform="twitter", username="@t", display_name="T", bio="bio",
        )
        provider.get_stats.return_value = MagicMock(followers=100, engagement_rate=0.05)
        provider.get_feed.return_value = []
        service = SocialMediaService(vault=vault)
        service._providers["twitter"] = provider
        result = service.scan_profiles()
        assert "twitter" in result
```

- [ ] **Step 2: Implement SocialMediaService facade**

The full facade in `src/homie_core/social_media/__init__.py` — follows exact same pattern as `src/homie_core/social/__init__.py` (SocialService). Iterate `_providers`, call provider methods, return dicts. `initialize()` loops over platform configs (twitter, reddit, linkedin, facebook, instagram, blog) checking vault for credentials.

- [ ] **Step 3: Run tests — should pass**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(social-media): add SocialMediaService facade"
```

---

## Chunk 3: Social Media Tools, Browser History, and Integration

### Task 14: Social Media AI Tools

**Files:**
- Create: `src/homie_core/social_media/tools.py`
- Test: `tests/unit/test_social_media/test_tools.py`

Register 9 tools: `sm_feed`, `sm_profile`, `sm_scan_profiles`, `sm_publish`, `sm_conversations`, `sm_dms`, `sm_send_dm`, `sm_search`, `sm_notifications`. All follow existing pattern — closure captures service, returns JSON string, `_truncate()` on output.

- [ ] **Step 1: Write failing tests** (test registration count + each tool returns valid JSON)
- [ ] **Step 2: Implement `register_social_media_tools()`**
- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat(social-media): add 9 AI tools — feed, profile, publish, DM, search, notifications"
```

---

### Task 15: Browser History Models

**Files:**
- Create: `src/homie_core/browser/__init__.py` (empty placeholder)
- Create: `src/homie_core/browser/models.py`
- Create: `tests/unit/test_browser/__init__.py`
- Create: `tests/unit/test_browser/test_models.py`

Models: `HistoryEntry`, `BrowsingPattern`, `BrowserConfig` — all with `to_dict()`.

- [ ] **Step 1-4:** Standard TDD cycle for dataclass models
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(browser): add data models — HistoryEntry, BrowsingPattern, BrowserConfig"
```

---

### Task 16: Browser History Readers

**Files:**
- Create: `src/homie_core/browser/readers.py`
- Test: `tests/unit/test_browser/test_readers.py`

Three readers: `ChromeReader`, `FirefoxReader`, `EdgeReader`. Each:
1. Copies browser's SQLite History file to temp location (browser locks it)
2. Reads `urls` and `visits` tables
3. Returns `list[HistoryEntry]`

- [ ] **Step 1: Write failing tests** (mock shutil.copy2, sqlite3.connect)

```python
# Key test pattern:
@patch("homie_core.browser.readers.sqlite3")
@patch("homie_core.browser.readers.shutil")
@patch("homie_core.browser.readers.Path")
def test_chrome_reader(self, MockPath, mock_shutil, mock_sqlite):
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        ("https://example.com", "Example", 13300000000000000, 5),  # Chrome timestamp format
    ]
    mock_sqlite.connect.return_value = mock_conn
    MockPath.return_value.exists.return_value = True

    reader = ChromeReader()
    entries = reader.read(since=0)
    assert len(entries) == 1
    assert entries[0].url == "https://example.com"
    assert entries[0].browser == "chrome"
```

- [ ] **Step 2: Implement readers**
- [ ] **Step 3: Run tests — should pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat(browser): add Chrome, Firefox, Edge history readers"
```

---

### Task 17: Browser Intelligence

**Files:**
- Create: `src/homie_core/browser/intelligence.py`
- Test: `tests/unit/test_browser/test_intelligence.py`

Analyzes history entries to produce `BrowsingPattern`: top domains, topics (via WebAnalyzer), peak hours, category breakdown.

- [ ] **Step 1-4:** TDD cycle
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(browser): add BrowsingIntelligence — pattern analysis from history"
```

---

### Task 18: BrowserHistoryService Facade

**Files:**
- Modify: `src/homie_core/browser/__init__.py`
- Test: `tests/unit/test_browser/test_service.py`

Facade methods: `initialize()`, `configure()`, `sync_tick()`, `get_history()`, `get_patterns()`, `scan()`, `get_config()`.

- [ ] **Step 1-4:** TDD cycle — same pattern as SocialService/FolderService
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(browser): add BrowserHistoryService facade"
```

---

### Task 19: Browser AI Tools

**Files:**
- Create: `src/homie_core/browser/tools.py`
- Test: `tests/unit/test_browser/test_tools.py`

Register 4 tools: `browser_history`, `browser_patterns`, `browser_scan`, `browser_config`.

- [ ] **Step 1-4:** TDD cycle
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(browser): add 4 AI tools — history, patterns, scan, config"
```

---

### Task 20: Daemon Integration

**Files:**
- Modify: `src/homie_app/daemon.py`

Add initialization for WebAnalyzer, SocialMediaService, BrowserHistoryService. Add tool registration in `_ensure_brain()`. Add sync callbacks.

- [ ] **Step 1: Add WebAnalyzer init** (after existing social service block)
- [ ] **Step 2: Add SocialMediaService init**
- [ ] **Step 3: Add BrowserHistoryService init**
- [ ] **Step 4: Add tool registration in `_ensure_brain()`**
- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: integrate social media, web analyzer, and browser history into daemon"
```

---

### Task 21: CLI Integration

**Files:**
- Modify: `src/homie_app/cli.py`

Add:
1. Connect commands for twitter, reddit, linkedin, facebook, blog
2. `homie sm` subparser with feed, profile, scan, publish, dms, send-dm
3. `homie browser` subparser with enable, disable, config, history, scan, patterns

- [ ] **Step 1: Add connect handlers** for each platform (same pattern as `_connect_slack`)
- [ ] **Step 2: Add `sm` subparser and `cmd_sm()` handler**
- [ ] **Step 3: Add `browser` subparser and `cmd_browser()` handler**
- [ ] **Step 4: Update CLI tests**
- [ ] **Step 5: Run full test suite**
- [ ] **Step 6: Commit**

```bash
git commit -m "feat: add CLI commands for social media (sm) and browser history"
```

---

### Task 22: Add feedparser dependency

**Files:**
- Modify: `pyproject.toml` or `requirements.txt`

- [ ] **Step 1: Add feedparser to dependencies**
- [ ] **Step 2: Install**

```bash
pip install feedparser
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: add feedparser dependency for blog RSS/Atom parsing"
```

---

### Task 23: Final Integration Test

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

- [ ] **Step 2: Verify all new modules import correctly**

```bash
python -c "from homie_core.social_media import SocialMediaService; from homie_core.web.analyzer import WebAnalyzer; from homie_core.browser import BrowserHistoryService; print('All imports OK')"
```

- [ ] **Step 3: Final commit if any fixes needed**
