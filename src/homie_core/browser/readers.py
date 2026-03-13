"""Browser history SQLite readers."""
from __future__ import annotations
import logging
import os
import shutil
import sqlite3
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from homie_core.browser.models import HistoryEntry

logger = logging.getLogger(__name__)

class BrowserReader(ABC):
    @abstractmethod
    def _db_path(self) -> Path: ...
    @abstractmethod
    def _query(self) -> str: ...
    @abstractmethod
    def _parse_row(self, row: tuple) -> HistoryEntry: ...
    @property
    def browser_name(self) -> str: return ""

    def read(self, since: float = 0) -> list[HistoryEntry]:
        db = self._db_path()
        if not db.exists():
            return []
        # Copy to temp (browser locks the file)
        tmp = Path(tempfile.mkdtemp()) / "history_copy"
        try:
            shutil.copy2(str(db), str(tmp))
            conn = sqlite3.connect(str(tmp))
            rows = conn.execute(self._query(), (since,)).fetchall()
            conn.close()
            return [self._parse_row(r) for r in rows]
        except Exception:
            logger.exception("Failed to read %s history", self.browser_name)
            return []
        finally:
            try:
                tmp.unlink(missing_ok=True)
                tmp.parent.rmdir()
            except Exception:
                pass

class ChromeReader(BrowserReader):
    browser_name = "chrome"
    def _db_path(self) -> Path:
        local = os.environ.get("LOCALAPPDATA", "")
        return Path(local) / "Google" / "Chrome" / "User Data" / "Default" / "History"
    def _query(self) -> str:
        return ("SELECT u.url, u.title, v.visit_time, v.visit_duration "
                "FROM urls u JOIN visits v ON u.id = v.url "
                "WHERE v.visit_time > ? ORDER BY v.visit_time DESC LIMIT 1000")
    def _parse_row(self, row: tuple) -> HistoryEntry:
        # Chrome stores time as microseconds since 1601-01-01
        chrome_epoch = 11644473600
        ts = (row[2] / 1_000_000) - chrome_epoch if row[2] else 0
        duration = row[3] / 1_000_000 if row[3] else None
        return HistoryEntry(url=row[0], title=row[1] or "", visit_time=ts,
                           duration=duration, browser="chrome")

class FirefoxReader(BrowserReader):
    browser_name = "firefox"
    def _db_path(self) -> Path:
        appdata = os.environ.get("APPDATA", "")
        profiles = Path(appdata) / "Mozilla" / "Firefox" / "Profiles"
        if profiles.exists():
            for p in profiles.iterdir():
                places = p / "places.sqlite"
                if places.exists():
                    return places
        return Path("nonexistent")
    def _query(self) -> str:
        return ("SELECT p.url, p.title, h.visit_date "
                "FROM moz_places p JOIN moz_historyvisits h ON p.id = h.place_id "
                "WHERE h.visit_date > ? ORDER BY h.visit_date DESC LIMIT 1000")
    def _parse_row(self, row: tuple) -> HistoryEntry:
        # Firefox stores time as microseconds since Unix epoch
        ts = row[2] / 1_000_000 if row[2] else 0
        return HistoryEntry(url=row[0], title=row[1] or "", visit_time=ts, browser="firefox")

class EdgeReader(BrowserReader):
    browser_name = "edge"
    def _db_path(self) -> Path:
        local = os.environ.get("LOCALAPPDATA", "")
        return Path(local) / "Microsoft" / "Edge" / "User Data" / "Default" / "History"
    def _query(self) -> str:
        return ChromeReader()._query()  # Edge uses same schema as Chrome
    def _parse_row(self, row: tuple) -> HistoryEntry:
        entry = ChromeReader()._parse_row(row)
        entry.browser = "edge"
        return entry
