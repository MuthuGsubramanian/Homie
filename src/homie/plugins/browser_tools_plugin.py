"""Browser tools plugin for HOMIE.

Gives HOMIE's local agent a small, safe browser harness inspired by the
Browser Tools SDK (https://libretto.sh/browser-tools): open a page, read its
visible text, list its links, and save snapshots for RAG ingestion -- all via
a local Playwright Chromium instance on the user's machine.

Network access is OPT-IN. The plugin stays dormant unless
``plugins.browser_tools.enabled`` is true in ``homie.config.yaml``, and every
navigation is validated against ``plugins.browser_tools.allowed_origins`` (a
list of URL prefixes). Nothing is uploaded anywhere; snapshots are kept in
memory and optionally written under ``~/.homie/browser``.

Requires the optional dependency ``playwright`` (``pip install playwright``
then ``playwright install chromium``). When Playwright is missing the plugin
activates in a degraded no-op mode and reports the missing dependency instead
of crashing HOMIE.

Example config::

    plugins:
      browser_tools:
        enabled: true
        allowed_origins:
          - "https://docs.python.org"
          - "https://news.ycombinator.com"
        headless: true
        timeout_ms: 15000
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from homie.config import HomieConfig, cfg_get

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 15_000
MAX_TEXT_CHARS = 20_000
MAX_LINKS = 100
SNAPSHOT_DIR = Path.home() / ".homie" / "browser"


class BrowserToolsError(Exception):
    """Raised when a browser tool call fails or is not permitted."""


@dataclass
class PageSnapshot:
    """Readable capture of a single page visit."""

    url: str
    title: str
    text: str
    links: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "links": self.links,
        }


class BrowserToolsPlugin:
    """Opt-in local browser harness exposing agent-callable tools."""

    name = "browser_tools"

    def __init__(self, cfg: HomieConfig) -> None:
        self.cfg = cfg
        self._playwright = None
        self._browser = None
        self._page = None
        self.active = False
        self.degraded_reason: Optional[str] = None
        self.last_snapshot: Optional[PageSnapshot] = None

    # -- lifecycle -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(cfg_get(self.cfg, "plugins", "browser_tools", "enabled", default=False))

    def activate(self) -> bool:
        """Start the plugin. Returns True when browser tools are usable."""
        if not self.enabled:
            LOGGER.info(
                "browser_tools disabled; set plugins.browser_tools.enabled to true to opt in"
            )
            return False
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.degraded_reason = (
                "playwright not installed (pip install playwright && playwright install chromium)"
            )
            LOGGER.warning("browser_tools degraded: %s", self.degraded_reason)
            return False

        headless = bool(cfg_get(self.cfg, "plugins", "browser_tools", "headless", default=True))
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless)
        self.active = True
        LOGGER.info("browser_tools active (headless=%s)", headless)
        return True

    def deactivate(self) -> None:
        """Stop the browser and release all resources."""
        for closer in (self._page, self._browser):
            try:
                if closer is not None:
                    closer.close()
            except Exception:  # noqa: BLE001
                LOGGER.debug("browser_tools: close failed", exc_info=True)
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:  # noqa: BLE001
                LOGGER.debug("browser_tools: playwright stop failed", exc_info=True)
        self._page = self._browser = self._playwright = None
        self.active = False
        LOGGER.info("browser_tools deactivated")

    # -- internals -----------------------------------------------------

    def _timeout_ms(self) -> int:
        return int(
            cfg_get(self.cfg, "plugins", "browser_tools", "timeout_ms", default=DEFAULT_TIMEOUT_MS)
        )

    def _allowed_origins(self) -> List[str]:
        return list(
            cfg_get(self.cfg, "plugins", "browser_tools", "allowed_origins", default=[]) or []
        )

    def _check_url(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise BrowserToolsError(f"Only http(s) URLs are allowed, got '{url}'")
        origins = self._allowed_origins()
        if not origins:
            raise BrowserToolsError(
                "No allowed_origins configured; add URL prefixes under "
                "plugins.browser_tools.allowed_origins"
            )
        if not any(url.startswith(prefix) for prefix in origins):
            raise BrowserToolsError(f"URL '{url}' is outside the configured allowed_origins")
        return url

    def _ensure_page(self):
        if not self.active or self._browser is None:
            raise BrowserToolsError(
                self.degraded_reason or "browser_tools is not active; call activate() first"
            )
        if self._page is None or self._page.is_closed():
            self._page = self._browser.new_page()
        return self._page

    # -- tools ---------------------------------------------------------

    def open_page(self, url: str) -> Dict[str, Any]:
        """Navigate to ``url`` and return a readable snapshot of the page."""
        page = self._ensure_page()
        target = self._check_url(url)
        page.goto(target, timeout=self._timeout_ms(), wait_until="domcontentloaded")

        text = page.inner_text("body")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()[:MAX_TEXT_CHARS]

        anchors = page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({text: e.innerText.trim(), href: e.href}))",
        )
        links = [a for a in anchors if a.get("text")][:MAX_LINKS]

        self.last_snapshot = PageSnapshot(url=page.url, title=page.title(), text=text, links=links)
        return self.last_snapshot.to_dict()

    def read_text(self) -> str:
        """Return the visible text of the most recently opened page."""
        if self.last_snapshot is None:
            raise BrowserToolsError("No page opened yet; call open_page first")
        return self.last_snapshot.text

    def list_links(self) -> List[Dict[str, str]]:
        """Return the labelled links of the most recently opened page."""
        if self.last_snapshot is None:
            raise BrowserToolsError("No page opened yet; call open_page first")
        return self.last_snapshot.links

    def save_snapshot(self, name: Optional[str] = None) -> str:
        """Persist the last snapshot as markdown under ``~/.homie/browser``.

        The saved file is plain markdown so it can be dropped straight into
        HOMIE's ChromaDB ingestion for RAG.
        """
        if self.last_snapshot is None:
            raise BrowserToolsError("No page opened yet; call open_page first")
        snap = self.last_snapshot
        slug = re.sub(r"[^a-z0-9]+", "-", (name or snap.title or "page").lower()).strip("-")[:60]
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = SNAPSHOT_DIR / f"{slug or 'page'}.md"
        body = f"# {snap.title}\n\nSource: {snap.url}\n\n{snap.text}\n"
        path.write_text(body, encoding="utf-8")
        LOGGER.info("browser_tools snapshot saved to %s", path)
        return str(path)

    def tools(self) -> Dict[str, Callable[..., Any]]:
        """Agent-callable tool registry, mirroring the Browser Tools SDK surface."""
        return {
            "browser_open": self.open_page,
            "browser_read_text": self.read_text,
            "browser_list_links": self.list_links,
            "browser_save_snapshot": self.save_snapshot,
        }


def register(cfg: HomieConfig) -> BrowserToolsPlugin:
    """Create and activate the plugin; safe to call even when disabled."""
    plugin = BrowserToolsPlugin(cfg)
    plugin.activate()
    return plugin


__all__ = [
    "BrowserToolsPlugin",
    "BrowserToolsError",
    "PageSnapshot",
    "register",
]
