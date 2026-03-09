import pytest
from pathlib import Path
from homie_core.storage.backup import BackupManager


@pytest.fixture
def source_dir(tmp_path):
    src = tmp_path / "homie_data"
    src.mkdir()
    (src / "homie.db").write_text("fake db content")
    (src / "config.yaml").write_text("llm:\n  backend: gguf")
    sub = src / "chroma"
    sub.mkdir()
    (sub / "index.bin").write_bytes(b"\x00\x01\x02")
    return src


def test_backup_and_restore(source_dir, tmp_path):
    backup_dir = tmp_path / "backup"
    restore_dir = tmp_path / "restored"
    mgr = BackupManager()
    mgr.create_backup(source_dir, backup_dir, passphrase="testpass123")
    assert (backup_dir / "homie_backup.enc").exists()
    mgr.restore_backup(backup_dir, restore_dir, passphrase="testpass123")
    assert (restore_dir / "homie.db").read_text() == "fake db content"
    assert (restore_dir / "chroma" / "index.bin").read_bytes() == b"\x00\x01\x02"


def test_wrong_passphrase_fails(source_dir, tmp_path):
    backup_dir = tmp_path / "backup"
    restore_dir = tmp_path / "restored"
    mgr = BackupManager()
    mgr.create_backup(source_dir, backup_dir, passphrase="correct")
    with pytest.raises(Exception):
        mgr.restore_backup(backup_dir, restore_dir, passphrase="wrong")
