from __future__ import annotations

import pytest

from homie_core.backend.state import StateBackend
from homie_core.backend.protocol import EditResult, FileContent, FileInfo, GrepMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_backend() -> StateBackend:
    return StateBackend()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class TestRead:
    def test_read_full_file(self):
        b = make_backend()
        b.write("/hello.txt", "line1\nline2\nline3\n")
        fc = b.read("/hello.txt")
        assert fc.content == "line1\nline2\nline3"
        assert fc.total_lines == 3
        assert fc.truncated is False

    def test_read_with_offset(self):
        b = make_backend()
        lines = "\n".join(f"line{i}" for i in range(1, 11)) + "\n"
        b.write("/multi.txt", lines)
        fc = b.read("/multi.txt", offset=2, limit=3)
        assert fc.content == "line3\nline4\nline5"
        assert fc.total_lines == 10
        assert fc.truncated is True

    def test_read_offset_beyond_end(self):
        b = make_backend()
        b.write("/short.txt", "only one line\n")
        fc = b.read("/short.txt", offset=100)
        assert fc.content == ""
        assert fc.total_lines == 1

    def test_read_nonexistent_raises(self):
        b = make_backend()
        with pytest.raises(FileNotFoundError):
            b.read("/nonexistent.txt")

    def test_read_limit_no_truncation(self):
        b = make_backend()
        b.write("/f.txt", "a\nb\nc\n")
        fc = b.read("/f.txt", offset=0, limit=100)
        assert fc.truncated is False
        assert fc.total_lines == 3


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

class TestWrite:
    def test_write_and_read_back(self):
        b = make_backend()
        b.write("/notes.txt", "hello world\n")
        fc = b.read("/notes.txt")
        assert "hello world" in fc.content

    def test_write_overwrites_existing(self):
        b = make_backend()
        b.write("/f.txt", "old\n")
        b.write("/f.txt", "new\n")
        fc = b.read("/f.txt")
        assert fc.content == "new"

    def test_write_nested_path(self):
        b = make_backend()
        b.write("/dir/nested.txt", "nested content\n")
        fc = b.read("/dir/nested.txt")
        assert "nested content" in fc.content

    def test_write_empty_content(self):
        b = make_backend()
        b.write("/empty.txt", "")
        fc = b.read("/empty.txt")
        assert fc.content == ""
        assert fc.total_lines == 0


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------

class TestEdit:
    def test_edit_success(self):
        b = make_backend()
        b.write("/code.py", "x = 1\ny = 2\n")
        result = b.edit("/code.py", "x = 1", "x = 99")
        assert result.success is True
        assert result.occurrences == 1
        fc = b.read("/code.py")
        assert "x = 99" in fc.content
        assert "x = 1" not in fc.content

    def test_edit_not_found(self):
        b = make_backend()
        b.write("/code.py", "x = 1\n")
        result = b.edit("/code.py", "z = 99", "z = 0")
        assert result.success is False
        assert result.occurrences == 0
        assert result.error is not None

    def test_edit_not_unique_returns_error(self):
        b = make_backend()
        b.write("/dup.py", "foo\nfoo\nfoo\n")
        result = b.edit("/dup.py", "foo", "bar")
        assert result.success is False
        assert result.occurrences == 3
        assert result.error is not None
        # File should be unchanged
        fc = b.read("/dup.py")
        assert fc.content == "foo\nfoo\nfoo"

    def test_edit_replace_all(self):
        b = make_backend()
        b.write("/dup.py", "foo\nfoo\nfoo\n")
        result = b.edit("/dup.py", "foo", "bar", replace_all=True)
        assert result.success is True
        fc = b.read("/dup.py")
        assert "foo" not in fc.content
        assert fc.content == "bar\nbar\nbar"

    def test_edit_file_not_found(self):
        b = make_backend()
        result = b.edit("/missing.py", "x", "y")
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# Ls
# ---------------------------------------------------------------------------

class TestLs:
    def test_ls_root_shows_files_and_dirs(self):
        b = make_backend()
        b.write("/file.txt", "content")
        b.write("/dir/nested.txt", "nested")
        entries = b.ls("/")
        names = {e.name for e in entries}
        assert "file.txt" in names
        assert "dir" in names

    def test_ls_root_dir_is_marked_is_dir(self):
        b = make_backend()
        b.write("/subdir/a.txt", "a")
        entries = b.ls("/")
        dir_entries = [e for e in entries if e.name == "subdir"]
        assert len(dir_entries) == 1
        assert dir_entries[0].is_dir is True

    def test_ls_root_file_not_marked_is_dir(self):
        b = make_backend()
        b.write("/readme.txt", "hi")
        entries = b.ls("/")
        file_entries = [e for e in entries if e.name == "readme.txt"]
        assert len(file_entries) == 1
        assert file_entries[0].is_dir is False

    def test_ls_subdir(self):
        b = make_backend()
        b.write("/mydir/a.txt", "a")
        b.write("/mydir/b.txt", "b")
        b.write("/other/c.txt", "c")
        entries = b.ls("/mydir")
        names = {e.name for e in entries}
        assert "a.txt" in names
        assert "b.txt" in names
        assert "c.txt" not in names

    def test_ls_subdir_shows_nested_dirs(self):
        b = make_backend()
        b.write("/top/mid/deep.txt", "deep")
        entries = b.ls("/top")
        names = {e.name for e in entries}
        assert "mid" in names
        dir_entries = [e for e in entries if e.name == "mid"]
        assert dir_entries[0].is_dir is True

    def test_ls_returns_fileinfo_instances(self):
        b = make_backend()
        b.write("/x.txt", "x")
        entries = b.ls("/")
        assert all(isinstance(e, FileInfo) for e in entries)

    def test_ls_empty_backend(self):
        b = make_backend()
        entries = b.ls("/")
        assert entries == []


# ---------------------------------------------------------------------------
# Glob
# ---------------------------------------------------------------------------

class TestGlob:
    def test_glob_star_txt(self):
        b = make_backend()
        b.write("/a.txt", "a")
        b.write("/b.py", "b")
        matches = b.glob("*.txt")
        assert any("a.txt" in m for m in matches)
        assert not any("b.py" in m for m in matches)

    def test_glob_double_star(self):
        b = make_backend()
        b.write("/top.txt", "top")
        b.write("/sub/nested.txt", "nested")
        matches = b.glob("**/*.txt")
        names = {m.rstrip("/").rsplit("/", 1)[-1] for m in matches}
        assert "top.txt" in names
        assert "nested.txt" in names

    def test_glob_no_match_returns_empty(self):
        b = make_backend()
        b.write("/a.py", "x")
        matches = b.glob("*.nonexistent")
        assert matches == []

    def test_glob_returns_strings(self):
        b = make_backend()
        b.write("/x.py", "x")
        matches = b.glob("*.py")
        assert all(isinstance(m, str) for m in matches)


# ---------------------------------------------------------------------------
# Grep
# ---------------------------------------------------------------------------

class TestGrep:
    def test_grep_literal_match(self):
        b = make_backend()
        b.write("/a.py", "def foo():\n    return 42\n")
        results = b.grep("return 42")
        assert len(results) == 1
        assert results[0].line_number == 2
        assert "return 42" in results[0].line

    def test_grep_regex_match(self):
        b = make_backend()
        b.write("/b.py", "x = 1\ny = 2\nz = 3\n")
        results = b.grep(r"[xyz] = \d")
        assert len(results) == 3

    def test_grep_no_match(self):
        b = make_backend()
        b.write("/c.py", "nothing here\n")
        results = b.grep("unicorn")
        assert results == []

    def test_grep_across_multiple_files(self):
        b = make_backend()
        b.write("/f1.py", "MARKER\n")
        b.write("/f2.py", "MARKER\n")
        results = b.grep("MARKER")
        assert len(results) == 2

    def test_grep_returns_grepmatches(self):
        b = make_backend()
        b.write("/target.py", "found_it\n")
        results = b.grep("found_it")
        assert len(results) == 1
        assert isinstance(results[0], GrepMatch)
        assert "target.py" in results[0].path

    def test_grep_with_include_filter(self):
        b = make_backend()
        b.write("/a.py", "MARKER\n")
        b.write("/b.txt", "MARKER\n")
        results = b.grep("MARKER", include="*.py")
        assert len(results) == 1
        assert "a.py" in results[0].path

    def test_grep_scoped_to_path(self):
        b = make_backend()
        b.write("/src/main.py", "MARKER\n")
        b.write("/tests/test_main.py", "MARKER\n")
        results = b.grep("MARKER", path="/src")
        assert len(results) == 1
        assert "src" in results[0].path
