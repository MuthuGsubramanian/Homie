"""Tests for git_scanner module."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from homie_core.context.git_scanner import (
    GitRepoStatus,
    _find_git_dirs,
    _get_repo_status,
    scan_git_repos,
)

# The Homie project root — guaranteed to be a git repo
HOMIE_ROOT = Path(__file__).parent.parent.parent.parent


class TestScanGitRepos:
    def test_finds_homie_repo(self):
        """scan_git_repos must find the Homie repo when given its parent dir."""
        parent = HOMIE_ROOT.parent
        results = scan_git_repos([parent], max_depth=1)
        paths = [Path(r.path) for r in results]
        assert HOMIE_ROOT in paths

    def test_returns_list_of_git_repo_status(self):
        results = scan_git_repos([HOMIE_ROOT.parent], max_depth=1)
        assert isinstance(results, list)
        assert all(isinstance(r, GitRepoStatus) for r in results)

    def test_nonexistent_root_is_skipped(self):
        results = scan_git_repos(["/nonexistent/path/that/does/not/exist"])
        assert results == []

    def test_empty_root_list(self):
        assert scan_git_repos([]) == []

    def test_max_depth_zero_finds_repo_at_root(self):
        """With max_depth=0 and the repo root itself, should find .git directly."""
        results = scan_git_repos([HOMIE_ROOT], max_depth=0)
        paths = [Path(r.path) for r in results]
        assert HOMIE_ROOT in paths

    def test_max_depth_respected(self, tmp_path):
        """Repos buried deeper than max_depth should not be found."""
        # Create a nested structure: tmp/level1/level2/level3/.git
        deep = tmp_path / "level1" / "level2" / "level3"
        deep.mkdir(parents=True)
        (deep / ".git").mkdir()
        # Create a fake HEAD so _get_repo_status returns None gracefully
        # (no real git repo, just checking _find_git_dirs depth)
        git_dirs = _find_git_dirs(tmp_path, max_depth=2)
        # .git is at depth 3 (tmp/level1/level2/level3/.git), should be excluded
        assert not any(d.parent == deep for d in git_dirs)

        git_dirs_deep = _find_git_dirs(tmp_path, max_depth=3)
        assert any(d.parent == deep for d in git_dirs_deep)


class TestGetRepoStatus:
    def test_returns_git_repo_status_for_valid_repo(self):
        result = _get_repo_status(HOMIE_ROOT)
        assert result is not None
        assert isinstance(result, GitRepoStatus)

    def test_branch_is_not_empty(self):
        result = _get_repo_status(HOMIE_ROOT)
        assert result is not None
        assert result.branch != ""

    def test_path_matches_input(self):
        result = _get_repo_status(HOMIE_ROOT)
        assert result is not None
        assert result.path == str(HOMIE_ROOT)

    def test_uncommitted_count_is_non_negative(self):
        result = _get_repo_status(HOMIE_ROOT)
        assert result is not None
        assert result.uncommitted_count >= 0

    def test_returns_none_for_non_git_dir(self, tmp_path):
        result = _get_repo_status(tmp_path)
        assert result is None

    def test_last_commit_msg_is_string(self):
        result = _get_repo_status(HOMIE_ROOT)
        assert result is not None
        assert isinstance(result.last_commit_msg, str)

    def test_last_commit_time_is_string(self):
        result = _get_repo_status(HOMIE_ROOT)
        assert result is not None
        assert isinstance(result.last_commit_time, str)


class TestFindGitDirs:
    def test_finds_git_dir_in_root(self):
        git_dirs = _find_git_dirs(HOMIE_ROOT, max_depth=0)
        assert any(d.name == ".git" for d in git_dirs)

    def test_returns_list(self):
        result = _find_git_dirs(HOMIE_ROOT, max_depth=1)
        assert isinstance(result, list)

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules" / "some_pkg"
        nm.mkdir(parents=True)
        (nm / ".git").mkdir()
        git_dirs = _find_git_dirs(tmp_path, max_depth=3)
        assert not any("node_modules" in str(d) for d in git_dirs)

    def test_skips_pycache(self, tmp_path):
        pc = tmp_path / "__pycache__" / "sub"
        pc.mkdir(parents=True)
        (pc / ".git").mkdir()
        git_dirs = _find_git_dirs(tmp_path, max_depth=3)
        assert not any("__pycache__" in str(d) for d in git_dirs)
