from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class WebPageAnalysis:
    url: str
    title: str
    page_type: str           # article, product, docs, social, video, webpage, non-html, error
    description: str
    main_content: str        # cleaned text (max 5000 chars)
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
