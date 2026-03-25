"""Tests for the CheckpointManager."""

from __future__ import annotations

from homie_core.finetune.training.checkpoint import CheckpointManager


class TestCheckpointManager:
    """CheckpointManager save / load / cleanup tests."""

    def test_save_and_load_state(self, tmp_path):
        mgr = CheckpointManager(base_dir=tmp_path, keep_recent=3)
        state = {"epoch": 2, "loss": 0.45, "step": 150}
        mgr.save(state, cycle=1)
        loaded = mgr.load(cycle=1)
        assert loaded == state

    def test_load_missing_cycle_returns_none(self, tmp_path):
        mgr = CheckpointManager(base_dir=tmp_path)
        assert mgr.load(cycle=99) is None

    def test_has_checkpoint(self, tmp_path):
        mgr = CheckpointManager(base_dir=tmp_path)
        assert mgr.has_checkpoint(cycle=1) is False
        mgr.save({"ok": True}, cycle=1)
        assert mgr.has_checkpoint(cycle=1) is True

    def test_cleanup_old_cycles(self, tmp_path):
        mgr = CheckpointManager(base_dir=tmp_path, keep_recent=2)
        for c in range(1, 6):
            mgr.save({"cycle": c}, cycle=c)

        mgr.cleanup()

        # Only the two most recent (4, 5) should survive
        assert not mgr.has_checkpoint(1)
        assert not mgr.has_checkpoint(2)
        assert not mgr.has_checkpoint(3)
        assert mgr.has_checkpoint(4)
        assert mgr.has_checkpoint(5)
