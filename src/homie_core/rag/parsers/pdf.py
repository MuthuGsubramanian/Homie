from __future__ import annotations

from pathlib import Path

from homie_core.rag.parsers import ParsedDocument, TextBlock, register_parser


@register_parser("pdf")
def parse_pdf(path: Path) -> ParsedDocument:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return ParsedDocument(source_path=str(path), metadata={"error": "PyMuPDF not installed"})
    doc = fitz.open(str(path))
    blocks = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        if text.strip():
            blocks.append(TextBlock(content=text.strip(), block_type="paragraph", page=page_num))
    metadata = {
        "format": "pdf",
        "pages": len(doc),
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
    }
    doc.close()
    return ParsedDocument(text_blocks=blocks, metadata=metadata, source_path=str(path))
