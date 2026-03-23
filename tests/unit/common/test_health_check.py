"""Tests for homie_core.common.health_check."""

import sqlite3
from pathlib import Path

import pytest

from homie_core.common.health_check import SystemHealthCheck


class TestCheckPythonVersion:
    def test_current_python_passes(self):
        hc = SystemHealthCheck()
        result = hc.check_python_version()
        assert result["passed"] is True
        assert result["check"] == "python_version"


class TestCheckDiskSpace:
    def test_reasonable_threshold_passes(self):
        hc = SystemHealthCheck()
        result = hc.check_disk_space(min_gb=0.001)
        assert result["passed"] is True

    def test_absurd_threshold_fails(self):
        hc = SystemHealthCheck()
        result = hc.check_disk_space(min_gb=999_999)
        assert result["passed"] is False
        assert "free" in result["message"]


class TestCheckModelFile:
    def test_existing_file(self, tmp_path: Path):
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 128)
        hc = SystemHealthCheck()
        result = hc.check_model_file(str(model))
        assert result["passed"] is True

    def test_missing_file(self, tmp_path: Path):
        hc = SystemHealthCheck()
        result = hc.check_model_file(str(tmp_path / "nope.bin"))
        assert result["passed"] is False

    def test_empty_file(self, tmp_path: Path):
        model = tmp_path / "empty.bin"
        model.write_bytes(b"")
        hc = SystemHealthCheck()
        result = hc.check_model_file(str(model))
        assert result["passed"] is False


class TestCheckDatabase:
    def test_writable_database(self, tmp_path: Path):
        db = tmp_path / "test.db"
        hc = SystemHealthCheck()
        result = hc.check_database(str(db))
        assert result["passed"] is True

    def test_invalid_path(self):
        hc = SystemHealthCheck()
        result = hc.check_database("/nonexistent/dir/test.db")
        assert result["passed"] is False


class TestCheckConfigValid:
    def test_empty_config_fails(self):
        hc = SystemHealthCheck(config={})
        result = hc.check_config_valid()
        assert result["passed"] is False

    def test_non_empty_config_passes(self):
        hc = SystemHealthCheck(config={"key": "value"})
        result = hc.check_config_valid()
        assert result["passed"] is True


class TestCheckOptionalDeps:
    def test_returns_result(self):
        hc = SystemHealthCheck()
        result = hc.check_optional_deps()
        # This check always passes — it just reports what's available.
        assert result["passed"] is True
        assert result["check"] == "optional_deps"


class TestRunAll:
    def test_returns_list_of_dicts(self):
        hc = SystemHealthCheck(config={"key": "val"})
        results = hc.run_all()
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert all("check" in r and "passed" in r for r in results)

    def test_includes_model_check_when_configured(self, tmp_path: Path):
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 64)
        hc = SystemHealthCheck(config={"model_path": str(model)})
        checks = {r["check"] for r in hc.run_all()}
        assert "model_file" in checks

    def test_includes_db_check_when_configured(self, tmp_path: Path):
        db = tmp_path / "test.db"
        hc = SystemHealthCheck(config={"db_path": str(db)})
        checks = {r["check"] for r in hc.run_all()}
        assert "database" in checks
