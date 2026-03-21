from __future__ import annotations
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class NightlyCleanup:
    """Enforces data retention policies and storage caps."""

    def __init__(self, storage_path: str | Path, retention_days: int = 30, max_storage_mb: int = 512):
        self._storage = Path(storage_path)
        self._retention_days = retention_days
        self._max_mb = max_storage_mb

    def run(
        self,
        screenshot_store=None,
        knowledge_graph=None,
    ) -> dict:
        """Run all cleanup tasks. Returns stats."""
        stats = {
            "screenshots_purged": 0,
            "graph_pruned": False,
            "storage_enforced": False,
        }

        # 1. Purge old screenshots
        if screenshot_store:
            try:
                stats["screenshots_purged"] = screenshot_store.purge_old()
            except Exception as e:
                logger.warning("Screenshot purge failed: %s", e)

        # 2. Decay and prune knowledge graph
        if knowledge_graph:
            try:
                knowledge_graph.decay_scores(half_life_days=self._retention_days // 2)
                knowledge_graph.prune(min_confidence=0.05)
                stats["graph_pruned"] = True
            except Exception as e:
                logger.warning("Graph cleanup failed: %s", e)

        # 3. Enforce storage cap
        try:
            stats["storage_enforced"] = self._enforce_storage_cap()
        except Exception as e:
            logger.warning("Storage cap enforcement failed: %s", e)

        return stats

    def _enforce_storage_cap(self) -> bool:
        if not self._storage.exists():
            return False
        usage = shutil.disk_usage(str(self._storage))
        used_mb = usage.used / (1024**2)
        if used_mb <= self._max_mb:
            return False
        # Delete oldest files in logs/ and conversation_history/ first
        for subdir in ["logs", "conversation_history", "large_tool_results"]:
            dir_path = self._storage / subdir
            if dir_path.exists():
                files = sorted(dir_path.iterdir(), key=lambda f: f.stat().st_mtime)
                for f in files:
                    if f.is_file():
                        f.unlink()
                        used_mb -= f.stat().st_size / (1024**2)
                        if used_mb <= self._max_mb * 0.8:
                            return True
        return True
