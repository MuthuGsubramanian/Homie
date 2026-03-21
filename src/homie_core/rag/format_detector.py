from __future__ import annotations
from enum import Enum
from pathlib import Path


class DocumentFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    HTML = "html"
    IMAGE = "image"
    CODE = "code"
    MARKDOWN = "markdown"
    EMAIL = "email"
    EPUB = "epub"
    CSV = "csv"
    TEXT = "text"
    UNKNOWN = "unknown"


# Magic bytes signatures
_MAGIC = {
    b"%PDF": DocumentFormat.PDF,
    b"PK": None,  # ZIP-based — check further (docx, xlsx, pptx, epub)
    b"\x89PNG": DocumentFormat.IMAGE,
    b"\xff\xd8\xff": DocumentFormat.IMAGE,
    b"GIF8": DocumentFormat.IMAGE,
    b"BM": DocumentFormat.IMAGE,
    b"II*\x00": DocumentFormat.IMAGE,  # TIFF
    b"MM\x00*": DocumentFormat.IMAGE,  # TIFF
}

_EXT_MAP = {
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
    ".xlsx": DocumentFormat.XLSX,
    ".xls": DocumentFormat.XLSX,
    ".pptx": DocumentFormat.PPTX,
    ".html": DocumentFormat.HTML, ".htm": DocumentFormat.HTML,
    ".jpg": DocumentFormat.IMAGE, ".jpeg": DocumentFormat.IMAGE,
    ".png": DocumentFormat.IMAGE, ".bmp": DocumentFormat.IMAGE,
    ".tiff": DocumentFormat.IMAGE, ".tif": DocumentFormat.IMAGE,
    ".gif": DocumentFormat.IMAGE, ".webp": DocumentFormat.IMAGE,
    ".md": DocumentFormat.MARKDOWN, ".mdx": DocumentFormat.MARKDOWN, ".rst": DocumentFormat.MARKDOWN,
    ".eml": DocumentFormat.EMAIL, ".msg": DocumentFormat.EMAIL,
    ".epub": DocumentFormat.EPUB,
    ".csv": DocumentFormat.CSV,
    ".txt": DocumentFormat.TEXT, ".log": DocumentFormat.TEXT,
}

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".c", ".cpp", ".h",
    ".java", ".rb", ".sh", ".bat", ".ps1", ".swift", ".kt", ".scala",
    ".r", ".sql", ".lua", ".php", ".pl", ".ex", ".exs", ".zig", ".nim",
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".xml", ".css",
    ".scss", ".less", ".vue", ".svelte",
}


def detect_format(path: Path) -> DocumentFormat:
    """Detect document format via magic bytes, then extension fallback."""
    # Try magic bytes first
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        for magic, fmt in _MAGIC.items():
            if header.startswith(magic):
                if fmt is not None:
                    return fmt
                # ZIP-based — check extension
                ext = path.suffix.lower()
                if ext == ".docx":
                    return DocumentFormat.DOCX
                elif ext == ".xlsx":
                    return DocumentFormat.XLSX
                elif ext == ".pptx":
                    return DocumentFormat.PPTX
                elif ext == ".epub":
                    return DocumentFormat.EPUB
                return DocumentFormat.UNKNOWN
    except OSError:
        return DocumentFormat.UNKNOWN

    # Extension fallback
    ext = path.suffix.lower()
    if ext in _CODE_EXTENSIONS:
        return DocumentFormat.CODE
    return _EXT_MAP.get(ext, DocumentFormat.UNKNOWN)
