"""Tests for homie_core.common.logging_config."""

import json
import logging
from pathlib import Path

import pytest

from homie_core.common.logging_config import get_logger, setup_logging


@pytest.fixture(autouse=True)
def _clean_root_logger():
    """Remove handlers added during tests so they don't leak."""
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    root.handlers = before


class TestSetupLogging:
    def test_creates_log_directory(self, tmp_path: Path):
        log_dir = tmp_path / "logs"
        setup_logging(log_dir)
        assert log_dir.exists()

    def test_creates_log_file(self, tmp_path: Path):
        setup_logging(tmp_path)
        assert (tmp_path / "homie.log").exists()

    def test_structured_json_output(self, tmp_path: Path):
        setup_logging(tmp_path, level="DEBUG", structured=True)
        logger = logging.getLogger("test_structured")
        logger.info("hello structured")
        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()
        content = (tmp_path / "homie.log").read_text(encoding="utf-8")
        # Each line should be valid JSON
        for line in content.strip().splitlines():
            record = json.loads(line)
            assert "level" in record
            assert "message" in record

    def test_plain_text_output(self, tmp_path: Path):
        setup_logging(tmp_path, structured=False)
        logger = logging.getLogger("test_plain")
        logger.warning("plain warning")
        for h in logging.getLogger().handlers:
            h.flush()
        content = (tmp_path / "homie.log").read_text(encoding="utf-8")
        assert "plain warning" in content

    def test_noisy_loggers_silenced(self, tmp_path: Path):
        setup_logging(tmp_path)
        assert logging.getLogger("urllib3").level >= logging.WARNING


class TestGetLogger:
    def test_auto_prefix(self):
        logger = get_logger("brain")
        assert logger.name == "homie.brain"

    def test_no_double_prefix(self):
        logger = get_logger("homie.brain")
        assert logger.name == "homie.brain"
