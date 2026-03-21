# tests/unit/self_healing/test_rollback.py
import pytest
from homie_core.self_healing.improvement.rollback import RollbackManager


class TestRollbackManager:
    def test_snapshot_and_rollback(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        target = tmp_path / "code.py"
        target.write_text("original")

        version_id = rm.snapshot(target, reason="testing")
        assert version_id is not None

        target.write_text("modified")
        rm.rollback(version_id)
        assert target.read_text() == "original"

    def test_snapshot_multiple_files(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("a_orig")
        f2.write_text("b_orig")

        vid = rm.snapshot([f1, f2], reason="multi")
        f1.write_text("a_mod")
        f2.write_text("b_mod")
        rm.rollback(vid)
        assert f1.read_text() == "a_orig"
        assert f2.read_text() == "b_orig"

    def test_rollback_nonexistent_version_raises(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        with pytest.raises(KeyError):
            rm.rollback("nonexistent")

    def test_evolution_log(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots", evolution_dir=tmp_path / "evolution")
        target = tmp_path / "code.py"
        target.write_text("original")
        vid = rm.snapshot(target, reason="test change")

        target.write_text("modified")
        rm.record_evolution(vid, diff="- original\n+ modified", reasoning="optimize", outcome="success")

        log = rm.get_evolution_log()
        assert len(log) == 1
        assert log[0]["version_id"] == vid
        assert log[0]["reasoning"] == "optimize"

    def test_blacklist_prevents_reattempt(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        rm.blacklist("bad_change_hash")
        assert rm.is_blacklisted("bad_change_hash")
        assert not rm.is_blacklisted("good_change_hash")
