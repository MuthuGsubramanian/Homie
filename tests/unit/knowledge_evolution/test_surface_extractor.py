# tests/unit/knowledge_evolution/test_surface_extractor.py
import pytest
from pathlib import Path
from homie_core.adaptive_learning.knowledge.intake.surface_extractor import SurfaceExtractor


class TestSurfaceExtractor:
    def test_extract_python_classes(self, tmp_path):
        code = tmp_path / "example.py"
        code.write_text("class MyClass:\n    def method(self):\n        pass\n\nclass Other:\n    pass\n")
        ext = SurfaceExtractor()
        result = ext.extract(code, file_type="python")
        assert "MyClass" in result["classes"]
        assert "Other" in result["classes"]

    def test_extract_python_functions(self, tmp_path):
        code = tmp_path / "funcs.py"
        code.write_text("def hello():\n    pass\n\ndef world(x, y):\n    return x + y\n")
        ext = SurfaceExtractor()
        result = ext.extract(code, file_type="python")
        assert "hello" in result["functions"]
        assert "world" in result["functions"]

    def test_extract_python_imports(self, tmp_path):
        code = tmp_path / "imports.py"
        code.write_text("import os\nfrom pathlib import Path\nimport json\n")
        ext = SurfaceExtractor()
        result = ext.extract(code, file_type="python")
        assert "os" in result["imports"]
        assert "pathlib" in result["imports"]

    def test_extract_markdown_headings(self, tmp_path):
        doc = tmp_path / "readme.md"
        doc.write_text("# Title\n## Section 1\nSome text.\n### Subsection\nMore text.\n")
        ext = SurfaceExtractor()
        result = ext.extract(doc, file_type="markdown")
        assert "Title" in result["headings"]
        assert "Section 1" in result["headings"]

    def test_extract_unknown_returns_minimal(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("some data here")
        ext = SurfaceExtractor()
        result = ext.extract(f, file_type="unknown")
        assert "line_count" in result

    def test_extract_returns_entities(self, tmp_path):
        code = tmp_path / "entities.py"
        code.write_text("class UserService:\n    '''Handles user operations.'''\n    pass\n")
        ext = SurfaceExtractor()
        result = ext.extract(code, file_type="python")
        assert len(result.get("entities", [])) >= 1
