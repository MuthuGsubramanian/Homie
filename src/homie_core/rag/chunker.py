"""Structure-aware document chunking.

Three strategies:
1. CodeChunker — splits by functions/classes, preserves signatures
2. MarkdownChunker — splits by headings, preserves section hierarchy
3. SlidingWindowChunker — generic fallback with configurable overlap

Each chunk carries metadata: source file, chunk type, line range,
parent section (for hierarchical formats).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Chunk:
    """A chunk of text with metadata for retrieval."""
    text: str
    source: str = ""           # file path or document ID
    chunk_type: str = "text"   # code_function, code_class, markdown_section, text
    start_line: int = 0
    end_line: int = 0
    parent_section: str = ""   # e.g. "## API Reference" for nested markdown
    language: str = ""         # programming language if code
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_search_text(self) -> str:
        """Text used for embedding/indexing — includes context prefix."""
        parts = []
        if self.source:
            parts.append(f"[{Path(self.source).name}]")
        if self.parent_section:
            parts.append(f"({self.parent_section})")
        parts.append(self.text)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Code chunker
# ---------------------------------------------------------------------------

# Patterns for top-level definitions in common languages
_PY_DEF = re.compile(r"^(class |def |async def )", re.MULTILINE)
_JS_DEF = re.compile(r"^(function |class |const \w+ = (?:async )?\(|export (?:default )?(?:function|class) )", re.MULTILINE)
_GO_DEF = re.compile(r"^(func |type )", re.MULTILINE)
_RUST_DEF = re.compile(r"^(fn |pub fn |impl |struct |enum |trait )", re.MULTILINE)

_LANG_PATTERNS = {
    ".py": _PY_DEF,
    ".js": _JS_DEF, ".ts": _JS_DEF, ".jsx": _JS_DEF, ".tsx": _JS_DEF,
    ".go": _GO_DEF,
    ".rs": _RUST_DEF,
}


def chunk_code(text: str, source: str = "", max_chunk: int = 1500, overlap: int = 100) -> list[Chunk]:
    """Split code by top-level definitions (functions, classes).

    Falls back to sliding window if no definitions found or for languages
    without explicit definition patterns.
    """
    ext = Path(source).suffix.lower() if source else ""
    pattern = _LANG_PATTERNS.get(ext)
    lines = text.split("\n")

    if not pattern:
        return _sliding_window_chunk(text, source, max_chunk, overlap, chunk_type="code")

    # Find definition boundaries
    boundaries = []
    for i, line in enumerate(lines):
        if pattern.match(line):
            boundaries.append(i)

    if not boundaries:
        return _sliding_window_chunk(text, source, max_chunk, overlap, chunk_type="code")

    # Add file end as final boundary
    boundaries.append(len(lines))

    chunks = []

    # Preamble (imports, module docstring) before first definition
    if boundaries[0] > 0:
        preamble = "\n".join(lines[:boundaries[0]]).strip()
        if preamble and len(preamble) > 20:
            chunks.append(Chunk(
                text=preamble,
                source=source,
                chunk_type="code_preamble",
                start_line=1,
                end_line=boundaries[0],
                language=ext.lstrip("."),
            ))

    # Each definition block
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        block = "\n".join(lines[start:end]).rstrip()

        if not block.strip():
            continue

        if len(block) > max_chunk:
            # Large block — sub-chunk with sliding window
            sub_chunks = _sliding_window_chunk(
                block, source, max_chunk, overlap,
                chunk_type="code_function",
                base_line=start + 1,
            )
            # First sub-chunk gets the definition signature as context
            for sc in sub_chunks:
                sc.language = ext.lstrip(".")
            chunks.extend(sub_chunks)
        else:
            # Determine type from first line
            first_line = lines[start].strip()
            ctype = "code_class" if first_line.startswith("class ") else "code_function"
            chunks.append(Chunk(
                text=block,
                source=source,
                chunk_type=ctype,
                start_line=start + 1,
                end_line=end,
                language=ext.lstrip("."),
            ))

    return chunks if chunks else _sliding_window_chunk(text, source, max_chunk, overlap, chunk_type="code")


# ---------------------------------------------------------------------------
# Markdown chunker
# ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)


def chunk_markdown(text: str, source: str = "", max_chunk: int = 1500, overlap: int = 100) -> list[Chunk]:
    """Split markdown by headings, preserving section hierarchy."""
    lines = text.split("\n")

    # Find heading positions
    headings: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    for i, line in enumerate(lines):
        m = _MD_HEADING.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append((i, level, title))

    if not headings:
        return _sliding_window_chunk(text, source, max_chunk, overlap, chunk_type="text")

    # Add document end
    headings.append((len(lines), 0, ""))

    chunks = []
    parent_stack: list[str] = []  # Track heading hierarchy

    # Content before first heading
    if headings[0][0] > 0:
        pre = "\n".join(lines[:headings[0][0]]).strip()
        if pre and len(pre) > 20:
            chunks.append(Chunk(
                text=pre,
                source=source,
                chunk_type="markdown_preamble",
                start_line=1,
                end_line=headings[0][0],
            ))

    for i in range(len(headings) - 1):
        line_idx, level, title = headings[i]
        next_idx = headings[i + 1][0]

        # Update parent stack
        while parent_stack and len(parent_stack) >= level:
            parent_stack.pop()
        parent_section = " > ".join(parent_stack) if parent_stack else ""
        parent_stack.append(title)

        section_text = "\n".join(lines[line_idx:next_idx]).rstrip()
        if not section_text.strip():
            continue

        if len(section_text) > max_chunk:
            sub_chunks = _sliding_window_chunk(
                section_text, source, max_chunk, overlap,
                chunk_type="markdown_section",
                base_line=line_idx + 1,
            )
            for sc in sub_chunks:
                sc.parent_section = f"{parent_section} > {title}" if parent_section else title
            chunks.extend(sub_chunks)
        else:
            chunks.append(Chunk(
                text=section_text,
                source=source,
                chunk_type="markdown_section",
                start_line=line_idx + 1,
                end_line=next_idx,
                parent_section=parent_section,
            ))

    return chunks if chunks else _sliding_window_chunk(text, source, max_chunk, overlap, chunk_type="text")


# ---------------------------------------------------------------------------
# Sliding window chunker (generic fallback)
# ---------------------------------------------------------------------------

def _sliding_window_chunk(
    text: str,
    source: str = "",
    max_chunk: int = 1500,
    overlap: int = 100,
    chunk_type: str = "text",
    base_line: int = 1,
) -> list[Chunk]:
    """Split text into overlapping chunks, breaking at line boundaries."""
    lines = text.split("\n")
    chunks = []
    current_lines: list[str] = []
    current_len = 0
    start_line = 0

    for i, line in enumerate(lines):
        line_len = len(line) + 1  # +1 for newline
        current_lines.append(line)
        current_len += line_len

        if current_len >= max_chunk:
            chunk_text = "\n".join(current_lines)
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_type=chunk_type,
                start_line=base_line + start_line,
                end_line=base_line + i,
            ))

            # Calculate overlap: keep last N characters worth of lines
            overlap_lines: list[str] = []
            overlap_len = 0
            for ol in reversed(current_lines):
                if overlap_len + len(ol) > overlap:
                    break
                overlap_lines.insert(0, ol)
                overlap_len += len(ol) + 1

            current_lines = overlap_lines
            current_len = overlap_len
            start_line = i - len(overlap_lines) + 1

    # Remainder
    if current_lines:
        chunk_text = "\n".join(current_lines)
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_type=chunk_type,
                start_line=base_line + start_line,
                end_line=base_line + len(lines) - 1,
            ))

    return chunks


# ---------------------------------------------------------------------------
# Auto-detect and chunk
# ---------------------------------------------------------------------------

_CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".c", ".cpp", ".h", ".java", ".rb", ".sh", ".bat"}
_MARKDOWN_EXTENSIONS = {".md", ".mdx", ".rst"}


def auto_chunk(text: str, source: str = "", max_chunk: int = 1500, overlap: int = 100) -> list[Chunk]:
    """Auto-detect document type and apply the best chunking strategy."""
    ext = Path(source).suffix.lower() if source else ""

    if ext in _CODE_EXTENSIONS:
        return chunk_code(text, source, max_chunk, overlap)
    elif ext in _MARKDOWN_EXTENSIONS:
        return chunk_markdown(text, source, max_chunk, overlap)
    else:
        return _sliding_window_chunk(text, source, max_chunk, overlap)
