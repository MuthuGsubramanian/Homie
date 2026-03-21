# tests/unit/self_healing/test_playbook.py
import json
import pytest
from homie_core.self_healing.recovery.playbook import RecoveryPlaybook, PlaybookEntry
from homie_core.self_healing.recovery.engine import RecoveryTier


class TestPlaybookEntry:
    def test_creation(self):
        entry = PlaybookEntry(
            module="inference",
            failure_type="timeout",
            tier=RecoveryTier.RETRY,
            action="retry with shorter max_tokens",
            success_count=0,
            fail_count=0,
        )
        assert entry.module == "inference"
        assert entry.success_rate == 0.0

    def test_success_rate(self):
        entry = PlaybookEntry(
            module="m", failure_type="f", tier=RecoveryTier.RETRY,
            action="a", success_count=7, fail_count=3,
        )
        assert entry.success_rate == pytest.approx(0.7)


class TestRecoveryPlaybook:
    def test_seed_playbook_has_entries(self):
        pb = RecoveryPlaybook()
        entries = pb.get_entries("inference")
        assert len(entries) > 0

    def test_get_best_strategy_for_failure(self):
        pb = RecoveryPlaybook()
        entry = pb.get_best_entry("inference", "timeout", RecoveryTier.RETRY)
        assert entry is not None
        assert entry.module == "inference"

    def test_record_outcome_updates_counts(self):
        pb = RecoveryPlaybook()
        entry = pb.get_best_entry("inference", "timeout", RecoveryTier.RETRY)
        assert entry is not None
        old_success = entry.success_count
        pb.record_outcome(entry, success=True)
        assert entry.success_count == old_success + 1

    def test_add_learned_entry(self):
        pb = RecoveryPlaybook()
        new_entry = PlaybookEntry(
            module="custom",
            failure_type="new_error",
            tier=RecoveryTier.FALLBACK,
            action="custom fix",
            success_count=0,
            fail_count=0,
            learned=True,
        )
        pb.add_entry(new_entry)
        entries = pb.get_entries("custom")
        assert any(e.action == "custom fix" for e in entries)

    def test_export_and_import(self, tmp_path):
        pb = RecoveryPlaybook()
        pb.record_outcome(pb.get_best_entry("inference", "timeout", RecoveryTier.RETRY), success=True)
        path = tmp_path / "playbook.json"
        pb.export_to_file(path)
        assert path.exists()

        pb2 = RecoveryPlaybook(seed=False)
        pb2.import_from_file(path)
        entries = pb2.get_entries("inference")
        assert len(entries) > 0
