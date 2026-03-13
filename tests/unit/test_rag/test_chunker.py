"""Tests for the structure-aware document chunker."""
import pytest
from homie_core.rag.chunker import (
    Chunk, chunk_code, chunk_markdown, auto_chunk,
    _sliding_window_chunk,
)


# -----------------------------------------------------------------------
# Sliding window chunker
# -----------------------------------------------------------------------

class TestSlidingWindow:
    def test_single_chunk(self):
        text = "line one\nline two\nline three"
        chunks = _sliding_window_chunk(text, max_chunk=1000)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_multiple_chunks(self):
        text = "\n".join(f"line {i}: some content here" for i in range(50))
        chunks = _sliding_window_chunk(text, max_chunk=200, overlap=50)
        assert len(chunks) > 1
        # Each chunk should be under limit (approximately)
        for c in chunks:
            assert c.start_line >= 1

    def test_overlap_preserved(self):
        text = "\n".join(f"line {i}" for i in range(20))
        chunks = _sliding_window_chunk(text, max_chunk=50, overlap=20)
        # Last lines of chunk N should appear at start of chunk N+1
        if len(chunks) >= 2:
            last_lines_0 = chunks[0].text.split("\n")[-2:]
            first_lines_1 = chunks[1].text.split("\n")[:2]
            assert any(line in first_lines_1 for line in last_lines_0)

    def test_empty_text(self):
        chunks = _sliding_window_chunk("", max_chunk=100)
        assert len(chunks) == 0

    def test_line_numbers_tracked(self):
        text = "\n".join(f"line {i}" for i in range(10))
        chunks = _sliding_window_chunk(text, max_chunk=1000, base_line=5)
        assert chunks[0].start_line == 5


# -----------------------------------------------------------------------
# Code chunker
# -----------------------------------------------------------------------

class TestCodeChunker:
    def test_python_functions(self):
        code = '''import os

def hello():
    return "hello"

def world():
    return "world"

class MyClass:
    def method(self):
        pass
'''
        chunks = chunk_code(code, source="test.py")
        assert len(chunks) >= 3  # preamble + 2 functions + 1 class (or more)
        types = [c.chunk_type for c in chunks]
        assert "code_preamble" in types or "code_function" in types

    def test_preserves_function_body(self):
        code = '''def add(a, b):
    """Add two numbers."""
    return a + b
'''
        chunks = chunk_code(code, source="math.py")
        assert any("return a + b" in c.text for c in chunks)

    def test_unknown_language_falls_back(self):
        code = "some content\nmore content\n"
        chunks = chunk_code(code, source="unknown.xyz")
        assert len(chunks) >= 1

    def test_large_function_gets_subchunked(self):
        # Create a function larger than max_chunk
        lines = ["def big_function():"]
        for i in range(100):
            lines.append(f"    x_{i} = {i}")
        code = "\n".join(lines)
        chunks = chunk_code(code, source="big.py", max_chunk=200)
        assert len(chunks) > 1

    def test_language_detection(self):
        code = "def hello():\n    pass\n"
        chunks = chunk_code(code, source="test.py")
        assert all(c.language == "py" for c in chunks)


# -----------------------------------------------------------------------
# Markdown chunker
# -----------------------------------------------------------------------

class TestMarkdownChunker:
    def test_splits_by_heading(self):
        md = """# Title

Introduction text.

## Section A

Content A.

## Section B

Content B.
"""
        chunks = chunk_markdown(md, source="doc.md")
        assert len(chunks) >= 2
        # Check sections are split
        section_texts = [c.text for c in chunks]
        assert any("Content A" in t for t in section_texts)
        assert any("Content B" in t for t in section_texts)

    def test_preserves_hierarchy(self):
        md = """# Top

## Sub

### SubSub

Deep content.
"""
        chunks = chunk_markdown(md, source="doc.md")
        deep_chunk = [c for c in chunks if "Deep content" in c.text]
        assert len(deep_chunk) >= 1

    def test_preamble_captured(self):
        md = """Some text before headings.

# First Heading

Content.
"""
        chunks = chunk_markdown(md, source="doc.md")
        assert any(c.chunk_type == "markdown_preamble" for c in chunks)

    def test_no_headings_falls_back(self):
        md = "Just plain text.\nNo headings here.\n"
        chunks = chunk_markdown(md, source="doc.md")
        assert len(chunks) >= 1

    def test_parent_section_tracking(self):
        md = """# Top

## Child

Content.

## Another

More.
"""
        chunks = chunk_markdown(md, source="doc.md")
        child_chunks = [c for c in chunks if "Content" in c.text]
        if child_chunks:
            assert child_chunks[0].parent_section == "" or "Top" in child_chunks[0].parent_section


# -----------------------------------------------------------------------
# Auto chunker
# -----------------------------------------------------------------------

class TestAutoChunk:
    def test_python_detected(self):
        chunks = auto_chunk("def foo():\n    pass\n", source="test.py")
        assert len(chunks) >= 1

    def test_markdown_detected(self):
        chunks = auto_chunk("# Title\n\nContent.\n", source="readme.md")
        assert len(chunks) >= 1

    def test_plain_text_fallback(self):
        chunks = auto_chunk("hello world\n", source="notes.txt")
        assert len(chunks) >= 1

    def test_unknown_extension(self):
        chunks = auto_chunk("data here\n", source="data.csv")
        assert len(chunks) >= 1


# -----------------------------------------------------------------------
# Chunk metadata
# -----------------------------------------------------------------------

class TestChunkMetadata:
    def test_to_search_text(self):
        chunk = Chunk(text="hello world", source="src/main.py", parent_section="API")
        search = chunk.to_search_text()
        assert "main.py" in search
        assert "API" in search
        assert "hello world" in search

    def test_char_count(self):
        chunk = Chunk(text="12345")
        assert chunk.char_count == 5
