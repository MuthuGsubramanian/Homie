"""Tests for NightlyCleanup."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestNightlyCleanupNoComponents:
    def test_run_with_no_components_returns_zero_stats(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path, retention_days=30, max_storage_mb=512)
        stats = cleanup.run()
        assert stats["screenshots_purged"] == 0
        assert stats["graph_pruned"] is False

    def test_run_returns_dict_with_expected_keys(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path)
        stats = cleanup.run()
        assert "screenshots_purged" in stats
        assert "graph_pruned" in stats
        assert "storage_enforced" in stats


class TestNightlyCleanupScreenshotStore:
    def test_run_calls_purge_old_on_screenshot_store(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path)
        mock_store = MagicMock()
        mock_store.purge_old.return_value = 5
        stats = cleanup.run(screenshot_store=mock_store)
        mock_store.purge_old.assert_called_once()
        assert stats["screenshots_purged"] == 5

    def test_run_reflects_purge_count_in_stats(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path)
        mock_store = MagicMock()
        mock_store.purge_old.return_value = 12
        stats = cleanup.run(screenshot_store=mock_store)
        assert stats["screenshots_purged"] == 12

    def test_screenshot_store_exception_does_not_crash(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path)
        mock_store = MagicMock()
        mock_store.purge_old.side_effect = RuntimeError("disk error")
        # Should not raise
        stats = cleanup.run(screenshot_store=mock_store)
        assert stats["screenshots_purged"] == 0


class TestNightlyCleanupKnowledgeGraph:
    def test_run_calls_decay_scores_and_prune(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path, retention_days=30)
        mock_graph = MagicMock()
        stats = cleanup.run(knowledge_graph=mock_graph)
        mock_graph.decay_scores.assert_called_once_with(half_life_days=15)
        mock_graph.prune.assert_called_once_with(min_confidence=0.05)
        assert stats["graph_pruned"] is True

    def test_half_life_is_half_of_retention_days(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path, retention_days=20)
        mock_graph = MagicMock()
        cleanup.run(knowledge_graph=mock_graph)
        mock_graph.decay_scores.assert_called_once_with(half_life_days=10)

    def test_graph_exception_does_not_crash(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path)
        mock_graph = MagicMock()
        mock_graph.decay_scores.side_effect = Exception("graph error")
        stats = cleanup.run(knowledge_graph=mock_graph)
        assert stats["graph_pruned"] is False


class TestNightlyCleanupIsolation:
    def test_screenshot_exception_does_not_prevent_graph_cleanup(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path)
        mock_store = MagicMock()
        mock_store.purge_old.side_effect = RuntimeError("store broken")
        mock_graph = MagicMock()
        stats = cleanup.run(screenshot_store=mock_store, knowledge_graph=mock_graph)
        mock_graph.decay_scores.assert_called_once()
        mock_graph.prune.assert_called_once()
        assert stats["graph_pruned"] is True

    def test_graph_exception_does_not_prevent_screenshot_cleanup(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path)
        mock_store = MagicMock()
        mock_store.purge_old.return_value = 3
        mock_graph = MagicMock()
        mock_graph.decay_scores.side_effect = Exception("graph broken")
        stats = cleanup.run(screenshot_store=mock_store, knowledge_graph=mock_graph)
        assert stats["screenshots_purged"] == 3
        assert stats["graph_pruned"] is False

    def test_storage_cap_not_enforced_when_under_limit(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        import shutil
        # Mock disk usage to be well under the cap
        fake_usage = shutil.disk_usage.__class__  # just for reference; we use a namedtuple mock
        with patch("homie_core.intelligence.nightly_cleanup.shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(used=100 * 1024**2)  # 100 MB used
            cleanup = NightlyCleanup(tmp_path, max_storage_mb=512)
            stats = cleanup.run()
        assert stats["storage_enforced"] is False

    def test_storage_enforced_returns_false_for_nonexistent_path(self, tmp_path):
        from homie_core.intelligence.nightly_cleanup import NightlyCleanup
        cleanup = NightlyCleanup(tmp_path / "nonexistent")
        result = cleanup._enforce_storage_cap()
        assert result is False
