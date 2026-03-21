from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import pytest


def test_pdf_magic_bytes(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake content")
    assert detect_format(f) == DocumentFormat.PDF


def test_docx_zip_plus_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "doc.docx"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("word/document.xml", "<root/>")
    assert detect_format(f) == DocumentFormat.DOCX


def test_xlsx_zip_plus_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "sheet.xlsx"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("xl/workbook.xml", "<root/>")
    assert detect_format(f) == DocumentFormat.XLSX


def test_pptx_zip_plus_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "slides.pptx"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("ppt/presentation.xml", "<root/>")
    assert detect_format(f) == DocumentFormat.PPTX


def test_png_magic_bytes(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    assert detect_format(f) == DocumentFormat.IMAGE


def test_jpeg_magic_bytes(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 12)
    assert detect_format(f) == DocumentFormat.IMAGE


def test_python_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "script.py"
    f.write_text("print('hello')")
    assert detect_format(f) == DocumentFormat.CODE


def test_js_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "app.js"
    f.write_text("console.log('hi')")
    assert detect_format(f) == DocumentFormat.CODE


def test_rust_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "main.rs"
    f.write_text("fn main() {}")
    assert detect_format(f) == DocumentFormat.CODE


def test_markdown_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "README.md"
    f.write_text("# Hello")
    assert detect_format(f) == DocumentFormat.MARKDOWN


def test_rst_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "docs.rst"
    f.write_text("Title\n=====")
    assert detect_format(f) == DocumentFormat.MARKDOWN


def test_unknown_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "file.xyz123"
    f.write_bytes(b"\x00\x01\x02\x03")
    assert detect_format(f) == DocumentFormat.UNKNOWN


def test_missing_file_returns_unknown():
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    result = detect_format(Path("/nonexistent/file.pdf"))
    assert result == DocumentFormat.UNKNOWN


def test_csv_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "data.csv"
    f.write_text("a,b,c\n1,2,3")
    assert detect_format(f) == DocumentFormat.CSV


def test_text_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "notes.txt"
    f.write_text("some text")
    assert detect_format(f) == DocumentFormat.TEXT


def test_html_extension(tmp_path):
    from homie_core.rag.format_detector import detect_format, DocumentFormat
    f = tmp_path / "page.html"
    f.write_text("<html></html>")
    assert detect_format(f) == DocumentFormat.HTML


def test_document_format_is_str_enum():
    from homie_core.rag.format_detector import DocumentFormat
    assert DocumentFormat.PDF == "pdf"
    assert DocumentFormat.CODE == "code"
