from __future__ import annotations

from pathlib import Path

from homie_core.rag.parsers import ParsedDocument, TextBlock, register_parser


@register_parser("html")
def parse_html(path: Path) -> ParsedDocument:
    content = path.read_text(encoding="utf-8", errors="replace")
    try:
        import trafilatura
        extracted = trafilatura.extract(content, include_tables=True, include_comments=False)
        if extracted:
            return ParsedDocument(
                text_blocks=[TextBlock(content=extracted, block_type="paragraph")],
                metadata={"format": "html"},
                source_path=str(path),
            )
    except ImportError:
        pass
    # Fallback: BeautifulSoup or strip tags
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
    except ImportError:
        import re
        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()
    return ParsedDocument(
        text_blocks=[TextBlock(content=text, block_type="paragraph")],
        metadata={"format": "html"},
        source_path=str(path),
    )
