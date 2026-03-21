from __future__ import annotations

import re
from pathlib import Path

from homie_core.rag.parsers import ParsedDocument, TextBlock, register_parser


@register_parser("epub")
def parse_epub(path: Path) -> ParsedDocument:
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        return ParsedDocument(source_path=str(path), metadata={"error": "ebooklib not installed"})
    book = epub.read_epub(str(path))
    blocks = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html = item.get_content().decode("utf-8", errors="replace")
        try:
            from bs4 import BeautifulSoup
            text = BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
        if text.strip():
            blocks.append(TextBlock(content=text.strip(), block_type="paragraph"))
    title_meta = book.get_metadata("DC", "title")
    title = title_meta[0][0] if title_meta else ""
    metadata = {"format": "epub", "title": title}
    return ParsedDocument(text_blocks=blocks, metadata=metadata, source_path=str(path))
