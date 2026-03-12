"""Browser history integration — readers, intelligence, and tools.

BrowserHistoryService is the main facade used by the daemon and CLI.
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any

from homie_core.browser.models import BrowserConfig, HistoryEntry, BrowsingPattern
from homie_core.browser.readers import ChromeReader, FirefoxReader, EdgeReader
from homie_core.browser.intelligence import BrowsingIntelligence

logger = logging.getLogger(__name__)

_READERS = {"chrome": ChromeReader, "firefox": FirefoxReader, "edge": EdgeReader}


class BrowserHistoryService:
    def __init__(self, vault, working_memory=None, web_analyzer=None):
        self._vault = vault
        self._working_memory = working_memory
        self._web_analyzer = web_analyzer
        self._config = BrowserConfig()
        self._intelligence = BrowsingIntelligence()
        self._last_sync: float = 0

    def initialize(self) -> dict:
        """Load config from vault. Returns config dict."""
        cred = self._vault.get_credential("browser", "config")
        if cred:
            try:
                data = json.loads(cred.access_token)
                self._config = BrowserConfig(
                    enabled=data.get("enabled", False),
                    browsers=data.get("browsers", ["chrome"]),
                    extension_enabled=data.get("extension_enabled", False),
                    exclude_domains=data.get("exclude_domains", []),
                    include_only_domains=data.get("include_only_domains", []),
                    retention_days=data.get("retention_days", 30),
                    analyze_urls=data.get("analyze_urls", True),
                )
            except Exception:
                logger.exception("Failed to load browser config")
        return self.get_config()

    def configure(self, **kwargs) -> dict:
        """Update config settings. Returns new config."""
        for key, value in kwargs.items():
            if hasattr(self._config, key) and value is not None:
                setattr(self._config, key, value)
        self._save_config()
        return self.get_config()

    def sync_tick(self) -> str:
        """Read new history since last sync."""
        if not self._config.enabled:
            return "Browser history disabled"
        entries = self._read_history(since=self._last_sync)
        self._last_sync = time.time()
        if entries and self._working_memory is not None:
            summaries = [{"url": e.url, "title": e.title, "browser": e.browser}
                         for e in entries[:10]]
            self._working_memory.update("browser_history", summaries)
        return f"Browser: {len(entries)} new page(s)"

    def get_history(self, limit: int = 50, domain: str | None = None,
                    since: float | None = None) -> list[dict]:
        """Get history entries with optional filters."""
        entries = self._read_history(since=since or 0)
        if domain:
            entries = [e for e in entries if domain.lower() in e.url.lower()]
        return [e.to_dict() for e in entries[:limit]]

    def get_patterns(self) -> dict:
        """Analyze browsing patterns from history."""
        entries = self._read_history(since=0)
        pattern = self._intelligence.analyze(entries)
        return pattern.to_dict()

    def scan(self) -> dict:
        """Full history scan + analysis."""
        entries = self._read_history(since=0)
        pattern = self._intelligence.analyze(entries)
        return {
            "entries_count": len(entries),
            "patterns": pattern.to_dict(),
        }

    def get_config(self) -> dict:
        return {
            "enabled": self._config.enabled,
            "browsers": self._config.browsers,
            "extension_enabled": self._config.extension_enabled,
            "exclude_domains": self._config.exclude_domains,
            "include_only_domains": self._config.include_only_domains,
            "retention_days": self._config.retention_days,
            "analyze_urls": self._config.analyze_urls,
        }

    def _read_history(self, since: float = 0) -> list[HistoryEntry]:
        entries: list[HistoryEntry] = []
        for browser_name in self._config.browsers:
            reader_cls = _READERS.get(browser_name)
            if not reader_cls:
                continue
            try:
                reader = reader_cls()
                browser_entries = reader.read(since=since)
                # Apply domain filters
                for entry in browser_entries:
                    domain = entry.url.split("/")[2] if len(entry.url.split("/")) > 2 else ""
                    if self._config.exclude_domains and any(
                        d in domain for d in self._config.exclude_domains
                    ):
                        continue
                    if self._config.include_only_domains and not any(
                        d in domain for d in self._config.include_only_domains
                    ):
                        continue
                    entries.append(entry)
            except Exception:
                logger.exception("Failed to read %s history", browser_name)
        return entries

    def _save_config(self) -> None:
        try:
            self._vault.store_credential(
                provider="browser", account_id="config",
                token_type="data", access_token=json.dumps(self.get_config()),
                refresh_token="", scopes=[],
            )
        except Exception:
            logger.exception("Failed to save browser config")
