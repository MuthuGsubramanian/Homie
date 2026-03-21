"""Tests for SessionPersistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from homie_core.knowledge.session_persistence import SessionPersistence
from homie_core.memory.working import WorkingMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wm(**kwargs) -> WorkingMemory:
    wm = WorkingMemory()
    for k, v in kwargs.items():
        wm.update(k, v)
    return wm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSessionPersistenceSave:
    def test_save_creates_json_file(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        wm = _make_wm(active_window="VSCode", flow_score=0.9)
        sp.save(wm)
        assert sp._path.exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        wm = _make_wm(task_description="fix bug", activity_type="coding")
        sp.save(wm)
        data = json.loads(sp._path.read_text())
        assert isinstance(data, dict)

    def test_saved_at_timestamp_present(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        wm = _make_wm(activity_type="writing")
        sp.save(wm)
        data = json.loads(sp._path.read_text())
        assert "saved_at" in data

    def test_known_keys_persisted(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        wm = _make_wm(
            active_window="Chrome",
            active_process="chrome.exe",
            activity_type="browsing",
            flow_score=0.7,
            task_description="research",
            sentiment="neutral",
        )
        sp.save(wm)
        data = json.loads(sp._path.read_text())
        assert data["active_window"] == "Chrome"
        assert data["activity_type"] == "browsing"
        assert data["flow_score"] == pytest.approx(0.7)
        assert data["task_description"] == "research"
        assert data["sentiment"] == "neutral"

    def test_conversation_summary_saved_not_full(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        wm = WorkingMemory()
        wm.add_message("user", "Hello Homie, can you help me?")
        wm.add_message("assistant", "Sure! What do you need?")
        sp.save(wm)
        data = json.loads(sp._path.read_text())
        # Should record count, not full messages list
        assert "last_conversation_length" in data
        assert data["last_conversation_length"] == 2
        # last_message should be truncated to 200 chars max
        assert "last_message" in data
        assert len(data["last_message"]) <= 200

    def test_creates_parent_directories(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c"
        sp = SessionPersistence(deep_path)
        wm = _make_wm(activity_type="coding")
        sp.save(wm)  # must not raise
        assert sp._path.exists()

    def test_none_values_not_saved(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        wm = WorkingMemory()  # nothing set
        sp.save(wm)
        data = json.loads(sp._path.read_text())
        # Only saved_at should be present
        for key in ["active_window", "active_process", "flow_score", "task_description"]:
            assert key not in data


class TestSessionPersistenceRestore:
    def test_restore_after_save_returns_state(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        wm_original = _make_wm(activity_type="coding", task_description="fix tests")
        sp.save(wm_original)

        wm_new = WorkingMemory()
        state = sp.restore(wm_new)

        assert state["activity_type"] == "coding"
        assert state["task_description"] == "fix tests"

    def test_restore_writes_keys_to_working_memory(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        sp.save(_make_wm(activity_type="writing", task_description="draft email"))

        wm = WorkingMemory()
        sp.restore(wm)

        assert wm.get("activity_type") == "writing"
        assert wm.get("task_description") == "draft email"

    def test_restore_no_file_returns_empty_dict(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        wm = WorkingMemory()
        state = sp.restore(wm)
        assert state == {}

    def test_restore_corrupt_file_returns_empty_dict(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        sp._path.parent.mkdir(parents=True, exist_ok=True)
        sp._path.write_text("not json {{{{")
        wm = WorkingMemory()
        state = sp.restore(wm)
        assert state == {}

    def test_restore_empty_file_returns_empty_dict(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        sp._path.parent.mkdir(parents=True, exist_ok=True)
        sp._path.write_text("")
        wm = WorkingMemory()
        state = sp.restore(wm)
        assert state == {}


class TestSessionPersistenceExists:
    def test_exists_false_before_save(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        assert not sp.exists()

    def test_exists_true_after_save(self, tmp_path):
        sp = SessionPersistence(tmp_path)
        sp.save(_make_wm(activity_type="test"))
        assert sp.exists()
