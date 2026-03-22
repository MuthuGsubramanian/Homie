# tests/unit/knowledge_evolution/test_scanner.py
import pytest
from homie_core.adaptive_learning.knowledge.intake.scanner import SourceScanner, FileInfo


class TestSourceScanner:
    def test_scan_directory(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "readme.md").write_text("# Readme")
        scanner = SourceScanner()
        files = scanner.scan_directory(tmp_path)
        assert len(files) == 3

    def test_file_info_has_metadata(self, tmp_path):
        (tmp_path / "test.py").write_text("x = 1")
        scanner = SourceScanner()
        files = scanner.scan_directory(tmp_path)
        assert files[0].path is not None
        assert files[0].file_type in ("python", "unknown")
        assert files[0].size_bytes > 0

    def test_filters_by_extension(self, tmp_path):
        (tmp_path / "code.py").write_text("x = 1")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.json").write_text("{}")
        scanner = SourceScanner(include_extensions={".py", ".json"})
        files = scanner.scan_directory(tmp_path)
        assert len(files) == 2

    def test_skips_hidden_and_venv(self, tmp_path):
        (tmp_path / "visible.py").write_text("x = 1")
        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "config").write_text("git")
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "pip.py").write_text("pip")
        scanner = SourceScanner()
        files = scanner.scan_directory(tmp_path)
        paths = [str(f.path) for f in files]
        assert not any(".git" in p for p in paths)
        assert not any(".venv" in p for p in paths)

    def test_respects_max_depth(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (tmp_path / "top.py").write_text("x")
        (deep / "deep.py").write_text("y")
        scanner = SourceScanner(max_depth=1)
        files = scanner.scan_directory(tmp_path)
        names = [f.path.name for f in files]
        assert "top.py" in names
        assert "deep.py" not in names
