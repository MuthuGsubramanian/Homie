import json
from pathlib import Path

from homie_core.intelligence.audit_log import AuditLogger


def test_log_query(tmp_path):
    logger = AuditLogger(log_dir=tmp_path)
    logger.log_query(prompt="What is the weather?", response="It's sunny.", model="gpt-4o")

    log_file = list(tmp_path.glob("audit_*.jsonl"))
    assert len(log_file) == 1

    lines = log_file[0].read_text().strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["prompt"] == "What is the weather?"
    assert entry["response"] == "It's sunny."
    assert entry["model"] == "gpt-4o"
    assert "timestamp" in entry


def test_multiple_entries_append(tmp_path):
    logger = AuditLogger(log_dir=tmp_path)
    logger.log_query("q1", "r1", "m1")
    logger.log_query("q2", "r2", "m2")

    log_file = list(tmp_path.glob("audit_*.jsonl"))
    lines = log_file[0].read_text().strip().split("\n")
    assert len(lines) == 2


def test_disabled_logger_does_nothing(tmp_path):
    logger = AuditLogger(log_dir=tmp_path, enabled=False)
    logger.log_query("q1", "r1", "m1")

    log_files = list(tmp_path.glob("audit_*.jsonl"))
    assert len(log_files) == 0
