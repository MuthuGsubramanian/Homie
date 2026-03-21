from __future__ import annotations

from pathlib import Path

from homie_core.rag.parsers import ParsedDocument, TextBlock, register_parser


@register_parser("image")
def parse_image(path: Path) -> ParsedDocument:
    try:
        import easyocr
    except ImportError:
        return ParsedDocument(
            source_path=str(path),
            metadata={"format": "image", "error": "easyocr not installed"},
        )
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(str(path))
    text = "\n".join(r[1] for r in results if r[1].strip())
    return ParsedDocument(
        text_blocks=[TextBlock(content=text, block_type="paragraph")] if text else [],
        metadata={"format": "image", "ocr_regions": len(results)},
        source_path=str(path),
    )
