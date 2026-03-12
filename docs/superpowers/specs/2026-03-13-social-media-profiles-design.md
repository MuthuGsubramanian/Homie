# Social Media Profile Integrations, URL Analysis & Browser History — Design Spec

**Date:** 2026-03-13
**Status:** Approved

## Overview

Add comprehensive social media integrations (Facebook, Instagram, LinkedIn, Reddit, Twitter/X, personal blog), a URL analysis engine, and a browser history module to Homie. Homie will monitor feeds, aggregate profiles, publish content, handle DMs, perform full profile scans to understand user social preferences, analyze any URL in detail, and intelligently learn from browser history.

**Architecture:** New `social_media` module with capability-based provider interfaces alongside existing `social` (Slack) module. Separate `web` module for URL analysis. Separate `browser` module for history reading and pattern analysis. All three integrate into the daemon and CLI.

**Coexistence with `social` module:** The existing `homie_core.social` module (Slack messaging) remains unchanged. The new `homie_core.social_media` module is entirely separate — different package, different service class, different tool prefix (`sm_` vs `social_`). Both register tools and sync callbacks independently in the daemon. There is no migration path needed; they serve different purposes (Slack workspace messaging vs social media profiles).

---

## 1. Capability Interfaces

Four abstract base classes that platforms mix and match, plus a base class for credential lifecycle:

```python
# src/homie_core/social_media/provider.py

class SocialMediaProviderBase:
    """Base class all providers inherit — handles credential lifecycle."""
    platform_name: str  # e.g. "twitter", "reddit"

    def connect(self, credential) -> bool:
        """Authenticate with stored credential. Returns True on success."""
        ...

    def refresh_token(self) -> bool:
        """Refresh OAuth token if expired. Returns True on success, False to re-auth."""
        ...

    @property
    def is_connected(self) -> bool: ...

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

Each concrete provider inherits `SocialMediaProviderBase` plus its capability ABCs. For example:
`class TwitterProvider(SocialMediaProviderBase, FeedProvider, ProfileProvider, PublishProvider, DirectMessageProvider)`

### Platform Capability Matrix

| Platform   | Feed | Profile | Publish | DM  |
|------------|------|---------|---------|-----|
| Twitter/X  | yes  | yes     | yes     | yes |
| Reddit     | yes  | yes     | yes     | yes |
| LinkedIn   | yes  | yes     | yes     | no (API restriction) |
| Facebook   | yes  | yes     | yes     | yes |
| Instagram  | yes  | yes     | yes* (business/creator accounts, requires app review) | yes* (business/creator accounts) |
| Blog       | yes  | yes** (feed metadata) | no      | no  |

\* Instagram Publish and DM require a business/creator account with Meta app review approval. The provider implements the interfaces but returns a clear error if the account type doesn't support the operation.

\** Blog `get_profile()` returns feed metadata (title, description, author, post count). `get_stats()` returns post count and feed update frequency.

---

## 2. Social Intelligence Layer

Cross-platform behavioral analysis engine:

```python
# src/homie_core/social_media/intelligence.py

@dataclass
class SocialProfile:
    platforms: dict[str, PlatformProfile]  # per-platform analysis
    cross_platform: CrossPlatformAnalysis

@dataclass
class PlatformProfile:
    topics: list[str]
    tone: str                    # casual, professional, technical, etc.
    avg_posts_per_week: float
    peak_hours: list[int]
    audience_size: int
    engagement_rate: float
    content_types: dict[str, float]  # {text: 0.6, image: 0.3, link: 0.1}

@dataclass
class CrossPlatformAnalysis:
    primary_topics: list[str]
    peak_hours: list[int]
    platform_preferences: dict[str, str]  # {tech_content: "twitter", professional: "linkedin"}
    audience_overlap: dict
    recommended_posting_times: dict[str, list[int]]

class SocialIntelligence:
    def __init__(self, vault): ...
    def analyze_profiles(self, providers: dict[str, SocialMediaProviderBase]) -> SocialProfile:
        """Receives connected providers from the facade, fetches profile data via
        ProfileProvider interface, analyzes patterns, stores result in vault."""
        ...
    def get_cached_profile(self) -> SocialProfile | None:
        """Load previously stored analysis from vault. Returns None if no scan done yet."""
        ...
```

`scan_profiles()` triggers this — fetches all connected profiles, analyzes posting patterns, tone, topics, peak hours, and audience. Result stored encrypted in vault.

---

## 3. Data Models

```python
# src/homie_core/social_media/models.py

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
    post_type: str = "text"       # text, image, video, link, poll

    def to_dict(self) -> dict: ...

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

    def to_dict(self) -> dict: ...

@dataclass
class ProfileStats:
    platform: str
    followers: int = 0
    following: int = 0
    post_count: int = 0
    engagement_rate: float = 0.0

    def to_dict(self) -> dict: ...

@dataclass
class Notification:
    id: str
    platform: str
    type: str                     # like, comment, mention, follow, share
    sender: str
    content: str
    timestamp: float
    url: str | None = None

    def to_dict(self) -> dict: ...

@dataclass
class Conversation:
    id: str
    platform: str
    participants: list[str]
    last_message_preview: str
    last_activity: float

    def to_dict(self) -> dict: ...

@dataclass
class DirectMessage:
    id: str
    platform: str
    conversation_id: str
    sender: str
    content: str
    timestamp: float

    def to_dict(self) -> dict: ...
```

---

## 4. Platform Providers

### Twitter/X (`twitter_provider.py`)
- **Capabilities:** Feed, Profile, Publish, DM
- **Auth:** OAuth 2.0 PKCE, port 8551
- **API:** Twitter API v2 via requests
- **Dependencies:** requests

### Reddit (`reddit_provider.py`)
- **Capabilities:** Feed, Profile, Publish, DM
- **Auth:** OAuth 2.0, port 8552
- **API:** Reddit JSON API via requests
- **Dependencies:** requests

### LinkedIn (`linkedin_provider.py`)
- **Capabilities:** Feed, Profile, Publish (no DM — API restriction)
- **Auth:** OAuth 2.0, port 8553
- **API:** LinkedIn REST API via requests
- **Dependencies:** requests

### Facebook (`facebook_provider.py`)
- **Capabilities:** Feed, Profile, Publish, DM
- **Auth:** OAuth 2.0, port 8554 (shared with Instagram — Meta Graph API)
- **API:** Meta Graph API via requests
- **Dependencies:** requests

### Instagram (`instagram_provider.py`)
- **Capabilities:** Feed, Profile, Publish, DM
- **Auth:** Shared with Facebook (Meta Graph API token)
- **API:** Instagram Graph API via requests
- **Dependencies:** requests

### Blog (`blog_provider.py`)
- **Capabilities:** Feed, Profile (read-only)
- **Auth:** None (RSS/Atom URL)
- **API:** RSS/Atom feed parsing
- **Dependencies:** requests, feedparser

---

## 5. File Structure

```
src/homie_core/social_media/
├── __init__.py              # SocialMediaService facade
├── models.py                # SocialPost, ProfileInfo, ProfileStats, etc.
├── provider.py              # FeedProvider, ProfileProvider, PublishProvider, DirectMessageProvider ABCs
├── intelligence.py          # SocialIntelligence — cross-platform behavioral analysis
├── oauth.py                 # Shared OAuth helpers (local redirect server factory)
├── tools.py                 # AI tool registrations (register_social_media_tools)
├── twitter_provider.py      # Feed, Profile, Publish, DM
├── reddit_provider.py       # Feed, Profile, Publish, DM
├── linkedin_provider.py     # Feed, Profile, Publish
├── facebook_provider.py     # Feed, Profile, Publish, DM
├── instagram_provider.py    # Feed, Profile, Publish, DM
└── blog_provider.py         # Feed, Profile (RSS/Atom via feedparser)

src/homie_core/web/
├── __init__.py
├── analyzer.py              # WebAnalyzer — fetch and analyze any URL
├── models.py                # WebPageAnalysis dataclass
└── tools.py                 # register_web_tools (web_analyze tool)

src/homie_core/browser/
├── __init__.py              # BrowserHistoryService facade
├── models.py                # HistoryEntry, BrowsingPattern, BrowserConfig
├── readers.py               # ChromeReader, FirefoxReader, EdgeReader
├── extension.py             # WebSocket server for real-time extension push
├── intelligence.py          # BrowsingIntelligence — pattern analysis
└── tools.py                 # register_browser_tools

tests/unit/test_social_media/
├── __init__.py
├── test_intelligence.py
├── test_service.py
├── test_tools.py
├── test_twitter_provider.py
├── test_reddit_provider.py
├── test_linkedin_provider.py
├── test_facebook_provider.py
├── test_instagram_provider.py
└── test_blog_provider.py

tests/unit/test_web/
├── __init__.py
├── test_analyzer.py
└── test_tools.py

tests/unit/test_browser/
├── __init__.py
├── test_readers.py
├── test_intelligence.py
├── test_service.py
└── test_tools.py
```

---

## 6. SocialMediaService Facade

```python
# src/homie_core/social_media/__init__.py

class SocialMediaService:
    def __init__(self, vault, working_memory=None):
        self._vault = vault
        self._working_memory = working_memory
        self._providers: dict[str, object] = {}
        self._intelligence = SocialIntelligence(vault)

    def initialize(self) -> list[str]:
        """Connect all platforms with stored credentials. Returns connected platform names."""

    def sync_tick(self) -> str:
        """Called by SyncManager — fetch new posts/notifications, update working memory."""

    def get_feed(self, platform: str = "all", limit: int = 20) -> list[dict]:
        """Get recent feed posts across platforms."""

    def get_profile(self, platform: str, username: str | None = None) -> dict:
        """Get profile info (own profile if username is None)."""

    def scan_profiles(self) -> dict:
        """Full scan of all connected profiles — triggers intelligence analysis."""

    def publish(self, platform: str, content: str, media_urls: list[str] | None = None) -> dict:
        """Publish a post to a platform. Returns {status, post_id, url}."""

    def get_conversations(self, platform: str, limit: int = 20) -> list[dict]:
        """List DM conversations on a platform."""

    def get_dms(self, platform: str, conversation_id: str, limit: int = 20) -> list[dict]:
        """Get messages in a DM conversation."""

    def send_dm(self, platform: str, recipient: str, text: str) -> dict:
        """Send a DM. Returns {status, message_id}."""

    def get_social_profile(self) -> dict:
        """Get the aggregated social intelligence profile."""

    def search(self, query: str, platform: str = "all", limit: int = 10) -> list[dict]:
        """Search posts across platforms."""

    def get_notifications(self, platform: str = "all", limit: int = 20) -> list[dict]:
        """Get recent notifications (likes, comments, mentions) across platforms."""
```

---

## 7. AI Tools

### Social Media Tools (`register_social_media_tools`)

| Tool Name | Description | Params |
|-----------|-------------|--------|
| `sm_feed` | Get recent feed posts | `platform="all"`, `limit="20"` |
| `sm_profile` | Get profile info | `platform`, `username=""` |
| `sm_scan_profiles` | Full profile scan + intelligence analysis | _(none)_ |
| `sm_publish` | Publish a post | `platform`, `content`, `media_urls=""` |
| `sm_conversations` | List DM conversations | `platform`, `limit="20"` |
| `sm_dms` | Get messages in a DM thread | `platform`, `conversation_id`, `limit="20"` |
| `sm_send_dm` | Send a direct message | `platform`, `recipient`, `text` |
| `sm_search` | Search posts across platforms | `query`, `platform="all"`, `limit="10"` |
| `sm_notifications` | Get recent notifications (likes, comments, mentions) | `platform="all"`, `limit="20"` |

### Web Tools (`register_web_tools`)

| Tool Name | Description | Params |
|-----------|-------------|--------|
| `web_analyze` | Fetch and analyze a webpage in detail | `url` |

### Browser Tools (`register_browser_tools`)

| Tool Name | Description | Params |
|-----------|-------------|--------|
| `browser_history` | Get browsing history | `limit="50"`, `domain=""`, `since=""` |
| `browser_patterns` | Get browsing pattern analysis | _(none)_ |
| `browser_scan` | Full history scan + analysis | _(none)_ |
| `browser_config` | View or update browser history settings | `enabled=""`, `browsers=""`, `exclude_domains=""`, `retention_days=""` |

All tools follow existing conventions: string params, `int()` try/except guards, `_truncate()` output, category-based grouping.

**`browser_config` param formats:** `enabled` accepts `"true"`/`"false"` (parsed with `str.lower() == "true"`), `browsers` accepts comma-separated list (e.g., `"chrome,firefox"`), `exclude_domains` accepts comma-separated domains, `retention_days` parsed with `int()`. Empty string means "don't change this setting."

---

## 8. Storage Schema

No new database tables. All data flows through existing vault API:

- **Credentials:** `vault.store_credential(provider, account_id, token_data)` — one entry per platform
- **Connection status:** `vault.set_connection_status(provider, connected, label, sync_interval)`
- **Intelligence profile:** `vault.store_credential("social_intelligence", "profile", encrypted_json)`
- **Browser config:** `vault.store_credential("browser", "config", config_json)`
- **Browser history entries:** Stored in cache DB (like folders module), auto-purged per retention setting

### Browser History Table Schema

```sql
CREATE TABLE IF NOT EXISTS browser_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    visit_time REAL NOT NULL,           -- Unix timestamp
    duration REAL,                       -- seconds on page (if available)
    browser TEXT NOT NULL,               -- chrome, firefox, edge
    domain TEXT NOT NULL,                -- extracted for fast filtering
    analyzed INTEGER NOT NULL DEFAULT 0  -- 1 if WebAnalyzer has processed this entry
);
CREATE INDEX IF NOT EXISTS idx_bh_visit_time ON browser_history(visit_time);
CREATE INDEX IF NOT EXISTS idx_bh_domain ON browser_history(domain);
```

Purge query: `DELETE FROM browser_history WHERE visit_time < ?` (threshold = now - retention_days * 86400).

Feed posts, DMs, and notifications are transient — fetched on demand, not cached locally.

### Intelligence Profile JSON Structure

```json
{
    "last_scan": "2026-03-13T10:00:00Z",
    "platforms": {
        "twitter": {"topics": ["tech", "AI"], "tone": "casual", "avg_posts_per_week": 12},
        "linkedin": {"topics": ["career", "leadership"], "tone": "professional", "avg_posts_per_week": 2}
    },
    "cross_platform": {
        "primary_topics": ["tech", "AI", "python"],
        "peak_hours": [9, 14, 21],
        "platform_preferences": {"tech_content": "twitter", "professional": "linkedin"},
        "audience_overlap": {}
    }
}
```

---

## 9. CLI Commands

### Social Media Connect

```
homie connect twitter      # OAuth flow -> port 8551
homie connect reddit       # OAuth flow -> port 8552
homie connect linkedin     # OAuth flow -> port 8553
homie connect facebook     # OAuth flow -> port 8554 (also covers Instagram)
homie connect blog <url>   # RSS/Atom URL — no OAuth needed
```

### Social Media Operations

```
homie sm feed [--platform twitter] [--limit 20]
homie sm profile [--platform twitter] [--username @someone]
homie sm scan                           # full profile scan + intelligence
homie sm publish <platform> <content>
homie sm dms <platform> [--conversation <id>]
homie sm send-dm <platform> <recipient> <text>
```

### Browser History

```
homie browser enable [--browsers chrome,firefox,edge]
homie browser disable
homie browser config [--exclude "bank.com,health.com"] [--retention 30]
homie browser history [--limit 50] [--domain github.com]
homie browser scan
homie browser patterns
```

---

## 10. URL Analysis Engine

```python
# src/homie_core/web/analyzer.py

class WebAnalyzer:
    def analyze_url(self, url: str) -> WebPageAnalysis:
        """Fetch URL and produce detailed content analysis."""
        # 1. Fetch with requests (timeout=10s, max 1MB response)
        # 2. Parse HTML with built-in html.parser
        # 3. Extract content using priority cascade:
        #    a. Look for <article> or <main> tag content
        #    b. Fall back to <body> with nav/header/footer/aside stripped
        #    c. Extract <title>, <meta name="description">, <meta property="og:*">
        #    d. Collect <h1>-<h6> headings, count <a> links and <img> images
        # 4. Classify page type from OG type, URL patterns, and content signals
        # 5. Truncate main_content to 5000 chars

# src/homie_core/web/models.py

@dataclass
class WebPageAnalysis:
    url: str
    title: str
    page_type: str               # article, product, docs, social, video, etc.
    description: str
    main_content: str            # cleaned text (max 5000 chars)
    headings: list[str]
    links_count: int
    images_count: int
    og_data: dict                # Open Graph metadata
    analyzed_at: float

    def to_dict(self) -> dict: ...
```

**Integration points:**
- **Chat:** Brain's `web_analyze` tool — user pastes URL, AI calls it automatically
- **Browser history:** `BrowserHistoryService` feeds URLs through analyzer for pattern building
- **Social media:** Feed posts containing URLs can optionally be analyzed inline

---

## 11. Browser History Module

### Data Models

```python
@dataclass
class HistoryEntry:
    url: str
    title: str
    visit_time: float
    duration: float | None
    browser: str                 # chrome, firefox, edge
    analysis: WebPageAnalysis | None  # lazy — analyzed on demand

@dataclass
class BrowsingPattern:
    top_domains: list[dict]      # {domain, visit_count, avg_duration}
    top_topics: list[str]
    peak_hours: list[int]
    daily_avg_pages: float
    category_breakdown: dict     # {news: 25%, dev_docs: 40%, social: 15%, ...}

@dataclass
class BrowserConfig:
    enabled: bool = False        # opt-in, off by default
    browsers: list[str] = field(default_factory=lambda: ["chrome"])
    extension_enabled: bool = False
    exclude_domains: list[str] = field(default_factory=list)
    include_only_domains: list[str] = field(default_factory=list)
    retention_days: int = 30
    analyze_urls: bool = True
```

### User Controls (Privacy-First)
- **Opt-in by default** — `enabled: False` until user explicitly enables
- **Browser selection** — user picks which browsers to read
- **Domain blocklist** — exclude sensitive domains (banking, health, etc.)
- **Domain allowlist** — optionally restrict to only specific domains
- **Retention** — auto-purge after N days
- **Pause/resume** — temporarily stop without losing config

### History Readers
- Each reader parses browser's SQLite history from default OS location
- Windows paths: `%LOCALAPPDATA%/Google/Chrome/User Data/Default/History`, etc.
- Read-only copy of DB file (browsers lock it) — copy to temp, read, delete
- Returns `list[HistoryEntry]`

### Browser Extension (Optional Upgrade)
- WebSocket server on port 8555
- Extension sends `{url, title, timestamp}` on each page visit
- Real-time feed — no SQLite polling needed
- Server-side ready now, extension build is a future deliverable

### BrowserHistoryService Facade

```python
class BrowserHistoryService:
    def __init__(self, vault, working_memory=None, web_analyzer=None): ...
    def initialize(self) -> dict: ...
    def configure(self, **kwargs) -> dict: ...
    def sync_tick(self) -> str: ...
    def get_history(self, limit=50, domain=None, since=None) -> list[dict]: ...
    def get_patterns(self) -> dict: ...
    def scan(self) -> dict: ...
    def get_config(self) -> dict: ...
```

---

## 12. Daemon Integration

### Initialization Order (in `start()`)

1. Web Analyzer (shared, no credentials needed)
2. Social Media Service (connect platforms from vault credentials)
3. Browser History Service (load config, start sync if enabled)

### Tool Registration (in `_ensure_brain()`)

```python
if self._web_analyzer:
    from homie_core.web.tools import register_web_tools
    register_web_tools(tool_registry, self._web_analyzer)
if self._social_media_service:
    from homie_core.social_media.tools import register_social_media_tools
    register_social_media_tools(tool_registry, self._social_media_service)
if self._browser_service:
    from homie_core.browser.tools import register_browser_tools
    register_browser_tools(tool_registry, self._browser_service)
```

### Sync Intervals
- Social media: 5 minutes (API rate limits)
- Browser history: 2 minutes (local file reads, cheap)
- Web analyzer: on-demand only (no sync tick)

### Rate Limiting and Error Handling
- `sync_tick()` iterates providers **sequentially** — a failure in one provider does not block others
- Each provider handles rate limit responses (HTTP 429) with exponential backoff (same pattern as `SlackProvider`)
- Token refresh is handled by `SocialMediaProviderBase.refresh_token()` — called automatically on 401 responses before retrying. If refresh fails, the provider is marked disconnected and the user is notified to re-authenticate via `homie connect <platform>`
- Rate limit errors are logged but not surfaced to the user unless persistent (3+ consecutive failures)

---

## Dependencies

- **requests** — already in project (HTTP client for all APIs)
- **feedparser** — new dependency (RSS/Atom parsing for blog provider)
- No other new dependencies. HTML parsing uses built-in `html.parser`.

## OAuth Port Assignments

| Platform        | Port | Notes |
|-----------------|------|-------|
| Email (Google)  | 8547 | Primary |
| Email (Google)  | 8548 | Alt/fallback (reserved) |
| Slack           | 8549 | |
| _(reserved)_    | 8550 | Available for future use |
| Twitter/X       | 8551 | |
| Reddit          | 8552 | |
| LinkedIn        | 8553 | |
| Facebook/Instagram | 8554 | Shared Meta Graph API |
| Browser Extension WS | 8555 | WebSocket, not HTTP redirect |
