"""WebAnalyzer — fetch and analyse web pages."""
from __future__ import annotations

import re
import time
from html.parser import HTMLParser
from typing import Any

import requests

from homie_core.web.models import WebPageAnalysis

_MAX_CONTENT = 5000
_SKIP_TAGS = frozenset({"nav", "footer", "header", "aside", "script", "style"})
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_CONTENT_TAGS = frozenset({"article", "main"})


class _ContentExtractor(HTMLParser):
    """Lightweight HTML parser that extracts structured data from a page."""

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.description = ""
        self.og_data: dict[str, str] = {}
        self.headings: list[str] = []
        self.links_count = 0
        self.images_count = 0

        # Internal state
        self._in_title = False
        self._title_parts: list[str] = []
        self._skip_depth = 0          # > 0 means inside a skipped tag
        self._content_depth = 0       # > 0 means inside article/main
        self._heading_parts: list[str] | None = None

        self._content_text: list[str] = []   # text inside article/main
        self._body_text: list[str] = []      # all body text (fallback)
        self._in_body = False

    # ------------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict: dict[str, str] = {k: (v or "") for k, v in attrs}

        if tag == "title":
            self._in_title = True
            return

        if tag == "meta":
            name = attr_dict.get("name", "").lower()
            prop = attr_dict.get("property", "").lower()
            content = attr_dict.get("content", "")
            if name == "description":
                self.description = content
            if prop.startswith("og:"):
                self.og_data[prop] = content
            return

        if tag == "body":
            self._in_body = True
            return

        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return

        if tag in _CONTENT_TAGS:
            self._content_depth += 1
            return

        if tag in _HEADING_TAGS and self._skip_depth == 0:
            self._heading_parts = []
            return

        if tag == "a":
            self.links_count += 1
        elif tag == "img":
            self.images_count += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            self.title = "".join(self._title_parts).strip()
            return

        if tag == "body":
            self._in_body = False
            return

        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return

        if tag in _CONTENT_TAGS and self._content_depth > 0:
            self._content_depth -= 1
            return

        if tag in _HEADING_TAGS and self._heading_parts is not None:
            self.headings.append("".join(self._heading_parts).strip())
            self._heading_parts = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
            return

        if self._skip_depth > 0:
            return

        if self._heading_parts is not None:
            self._heading_parts.append(data)

        if self._content_depth > 0:
            self._content_text.append(data)

        if self._in_body:
            self._body_text.append(data)

    # ------------------------------------------------------------------
    @property
    def main_content(self) -> str:
        """Return cleaned main content (prefer article/main, fallback to body)."""
        raw = " ".join(self._content_text) if self._content_text else " ".join(self._body_text)
        text = re.sub(r"\s+", " ", raw).strip()
        return text[:_MAX_CONTENT]


# Page-type classification helpers ----------------------------------------

_URL_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"youtube\.com|vimeo\.com|youtu\.be", "video"),
    (r"twitter\.com|x\.com|facebook\.com|instagram\.com|reddit\.com", "social"),
    (r"amazon\.|ebay\.|shopify\.|etsy\.", "product"),
    (r"docs\.|readthedocs|wiki|documentation", "docs"),
]


def _classify_page(og_data: dict[str, str], url: str) -> str:
    og_type = og_data.get("og:type", "").lower()
    if og_type:
        if og_type == "article":
            return "article"
        if og_type in ("product", "product.item"):
            return "product"
        if "video" in og_type:
            return "video"

    for pattern, ptype in _URL_TYPE_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return ptype

    return "webpage"


# Public API --------------------------------------------------------------

class WebAnalyzer:
    """Fetches and analyses web pages."""

    def analyze_url(self, url: str) -> WebPageAnalysis:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Homie/1.0"})
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "html" not in content_type.lower():
                return WebPageAnalysis(
                    url=url,
                    title="",
                    page_type="non-html",
                    description=f"Content-Type: {content_type}",
                    main_content="",
                    headings=[],
                    links_count=0,
                    images_count=0,
                    og_data={},
                    analyzed_at=time.time(),
                )

            extractor = _ContentExtractor()
            extractor.feed(resp.text)

            page_type = _classify_page(extractor.og_data, url)

            return WebPageAnalysis(
                url=url,
                title=extractor.title,
                page_type=page_type,
                description=extractor.description,
                main_content=extractor.main_content,
                headings=extractor.headings,
                links_count=extractor.links_count,
                images_count=extractor.images_count,
                og_data=extractor.og_data,
                analyzed_at=time.time(),
            )

        except Exception as exc:
            return WebPageAnalysis(
                url=url,
                title="",
                page_type="error",
                description=str(exc),
                main_content="",
                headings=[],
                links_count=0,
                images_count=0,
                og_data={},
                analyzed_at=time.time(),
            )
