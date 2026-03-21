# tests/unit/self_healing/test_patcher.py
import pytest
from homie_core.self_healing.improvement.patcher import CodePatcher
from homie_core.self_healing.improvement.rollback import RollbackManager


class TestCodePatcher:
    def test_apply_patch_modifies_file(self, tmp_path):
        target = tmp_path / "module.py"
        target.write_text("def slow():\n    x = 1\n    return x\n")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        patcher = CodePatcher(rollback_manager=rm, project_root=tmp_path)
        version_id = patcher.apply_patch(
            file_path=target,
            old_text="    x = 1\n    return x",
            new_text="    return 1  # optimized",
            reason="remove unnecessary variable",
        )
        assert version_id is not None
        assert "return 1  # optimized" in target.read_text()

    def test_apply_patch_creates_snapshot(self, tmp_path):
        target = tmp_path / "module.py"
        target.write_text("original content")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        patcher = CodePatcher(rollback_manager=rm, project_root=tmp_path)
        version_id = patcher.apply_patch(
            file_path=target,
            old_text="original content",
            new_text="modified content",
            reason="test",
        )
        # Rollback should restore
        rm.rollback(version_id)
        assert target.read_text() == "original content"

    def test_apply_patch_rejects_locked_file(self, tmp_path):
        target = tmp_path / "security" / "vault.py"
        target.parent.mkdir()
        target.write_text("secret")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        patcher = CodePatcher(rollback_manager=rm, project_root=tmp_path, locked_paths=["security/"])
        with pytest.raises(PermissionError):
            patcher.apply_patch(
                file_path=target,
                old_text="secret",
                new_text="hacked",
                reason="evil",
            )

    def test_apply_patch_fails_if_old_text_not_found(self, tmp_path):
        target = tmp_path / "module.py"
        target.write_text("actual content")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        patcher = CodePatcher(rollback_manager=rm, project_root=tmp_path)
        with pytest.raises(ValueError, match="not found"):
            patcher.apply_patch(
                file_path=target,
                old_text="nonexistent text",
                new_text="replacement",
                reason="test",
            )
