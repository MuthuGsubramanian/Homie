from __future__ import annotations

from pathlib import Path

from homie_core.rag.parsers import ParsedDocument, TextBlock, register_parser

_EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".java": "java",
    ".rb": "ruby",
    ".sh": "bash",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sql": "sql",
    ".php": "php",
    ".lua": "lua",
}


def _parse_with_tree_sitter(path: Path, content: str, language: str) -> ParsedDocument:
    import tree_sitter  # noqa: F401 — just check it's importable
    # Full tree-sitter integration requires language-specific grammars installed separately.
    # Fall through so the caller uses the plain-text fallback.
    raise ImportError("tree-sitter grammar not configured")


@register_parser("code")
def parse_code(path: Path) -> ParsedDocument:
    content = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()
    language = _EXT_TO_LANG.get(ext, "unknown")
    # Try tree-sitter for AST-aware parsing
    try:
        return _parse_with_tree_sitter(path, content, language)
    except (ImportError, Exception):
        pass
    # Fallback: treat as text with language metadata
    return ParsedDocument(
        text_blocks=[TextBlock(content=content, block_type="code", language=language)],
        metadata={"format": "code", "language": language, "lines": content.count("\n") + 1},
        source_path=str(path),
    )
