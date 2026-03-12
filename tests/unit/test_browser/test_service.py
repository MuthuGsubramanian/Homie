"""Tests for BrowserHistoryService."""
import json
import pytest
from unittest.mock import MagicMock, patch

from homie_core.browser import BrowserHistoryService
from homie_core.browser.models import BrowserConfig, HistoryEntry


class TestBrowserHistoryService:
    def _make_vault(self, config_data=None):
        vault = MagicMock()
        if config_data:
            cred = MagicMock()
            cred.access_token = json.dumps(config_data)
            vault.get_credential.return_value = cred
        else:
            vault.get_credential.return_value = None
        return vault

    def _make_mock_reader(self, entries):
        mock_reader_cls = MagicMock()
        mock_reader = MagicMock()
        mock_reader.read.return_value = entries
        mock_reader_cls.return_value = mock_reader
        return mock_reader_cls

    def test_initialize_no_config(self):
        vault = self._make_vault()
        service = BrowserHistoryService(vault=vault)
        config = service.initialize()
        assert config["enabled"] is False
        assert config["browsers"] == ["chrome"]

    def test_initialize_with_config(self):
        vault = self._make_vault({"enabled": True, "browsers": ["firefox"], "retention_days": 7})
        service = BrowserHistoryService(vault=vault)
        config = service.initialize()
        assert config["enabled"] is True
        assert config["browsers"] == ["firefox"]
        assert config["retention_days"] == 7

    def test_initialize_corrupt_config(self):
        vault = MagicMock()
        cred = MagicMock()
        cred.access_token = "not-json"
        vault.get_credential.return_value = cred
        service = BrowserHistoryService(vault=vault)
        config = service.initialize()
        # Falls back to defaults
        assert config["enabled"] is False

    def test_configure(self):
        vault = self._make_vault()
        service = BrowserHistoryService(vault=vault)
        service.initialize()
        config = service.configure(enabled=True, retention_days=14)
        assert config["enabled"] is True
        assert config["retention_days"] == 14
        # Verify save was called
        vault.store_credential.assert_called_once()

    def test_configure_ignores_none(self):
        vault = self._make_vault()
        service = BrowserHistoryService(vault=vault)
        service.initialize()
        config = service.configure(enabled=None)
        assert config["enabled"] is False  # unchanged

    def test_get_config(self):
        vault = self._make_vault()
        service = BrowserHistoryService(vault=vault)
        service.initialize()
        config = service.get_config()
        assert "enabled" in config
        assert "browsers" in config
        assert "exclude_domains" in config
        assert "retention_days" in config
        assert "analyze_urls" in config

    def test_sync_tick_disabled(self):
        vault = self._make_vault()
        service = BrowserHistoryService(vault=vault)
        service.initialize()
        result = service.sync_tick()
        assert result == "Browser history disabled"

    def test_sync_tick_enabled(self):
        vault = self._make_vault({"enabled": True, "browsers": ["chrome"]})
        mock_reader_cls = self._make_mock_reader([
            HistoryEntry(url="https://example.com", title="Ex", visit_time=1700000000.0),
        ])

        working_memory = MagicMock()
        service = BrowserHistoryService(vault=vault, working_memory=working_memory)
        service.initialize()
        with patch.dict("homie_core.browser._READERS", {"chrome": mock_reader_cls}):
            result = service.sync_tick()
        assert "1 new page" in result
        working_memory.update.assert_called_once()

    def test_get_history(self):
        vault = self._make_vault({"enabled": True, "browsers": ["chrome"]})
        mock_reader_cls = self._make_mock_reader([
            HistoryEntry(url="https://github.com/repo", title="Repo", visit_time=1700000000.0),
            HistoryEntry(url="https://example.com", title="Ex", visit_time=1700001000.0),
        ])

        service = BrowserHistoryService(vault=vault)
        service.initialize()
        with patch.dict("homie_core.browser._READERS", {"chrome": mock_reader_cls}):
            results = service.get_history(limit=10)
        assert len(results) == 2
        assert results[0]["url"] == "https://github.com/repo"

    def test_get_history_domain_filter(self):
        vault = self._make_vault({"enabled": True, "browsers": ["chrome"]})
        mock_reader_cls = self._make_mock_reader([
            HistoryEntry(url="https://github.com/repo", title="Repo", visit_time=1700000000.0),
            HistoryEntry(url="https://example.com", title="Ex", visit_time=1700001000.0),
        ])

        service = BrowserHistoryService(vault=vault)
        service.initialize()
        with patch.dict("homie_core.browser._READERS", {"chrome": mock_reader_cls}):
            results = service.get_history(domain="github")
        assert len(results) == 1
        assert "github" in results[0]["url"]

    def test_get_patterns(self):
        vault = self._make_vault({"enabled": True, "browsers": ["chrome"]})
        mock_reader_cls = self._make_mock_reader([
            HistoryEntry(url="https://github.com/repo", title="Repo", visit_time=1700000000.0),
        ])

        service = BrowserHistoryService(vault=vault)
        service.initialize()
        with patch.dict("homie_core.browser._READERS", {"chrome": mock_reader_cls}):
            patterns = service.get_patterns()
        assert "top_domains" in patterns
        assert "category_breakdown" in patterns

    def test_scan(self):
        vault = self._make_vault({"enabled": True, "browsers": ["chrome"]})
        mock_reader_cls = self._make_mock_reader([
            HistoryEntry(url="https://github.com/repo", title="Repo", visit_time=1700000000.0),
        ])

        service = BrowserHistoryService(vault=vault)
        service.initialize()
        with patch.dict("homie_core.browser._READERS", {"chrome": mock_reader_cls}):
            result = service.scan()
        assert result["entries_count"] == 1
        assert "patterns" in result

    def test_exclude_domains(self):
        vault = self._make_vault({
            "enabled": True, "browsers": ["chrome"],
            "exclude_domains": ["example.com"],
        })
        mock_reader_cls = self._make_mock_reader([
            HistoryEntry(url="https://github.com/repo", title="Repo", visit_time=1700000000.0),
            HistoryEntry(url="https://example.com/page", title="Ex", visit_time=1700001000.0),
        ])

        service = BrowserHistoryService(vault=vault)
        service.initialize()
        with patch.dict("homie_core.browser._READERS", {"chrome": mock_reader_cls}):
            results = service.get_history()
        assert len(results) == 1
        assert "github" in results[0]["url"]

    def test_save_config_error_handled(self):
        vault = self._make_vault()
        vault.store_credential.side_effect = Exception("DB error")
        service = BrowserHistoryService(vault=vault)
        service.initialize()
        # Should not raise
        config = service.configure(enabled=True)
        assert config["enabled"] is True
