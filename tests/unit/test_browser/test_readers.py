"""Tests for browser history readers."""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

from homie_core.browser.readers import ChromeReader, FirefoxReader, EdgeReader
from homie_core.browser.models import HistoryEntry


class TestChromeReader:
    @patch("homie_core.browser.readers.sqlite3")
    @patch("homie_core.browser.readers.shutil")
    def test_read_success(self, mock_shutil, mock_sqlite):
        mock_conn = MagicMock()
        # Chrome epoch: microseconds since 1601-01-01
        # For a known unix timestamp ~1700000000, chrome_time = (1700000000 + 11644473600) * 1_000_000
        chrome_time = (1700000000 + 11644473600) * 1_000_000
        mock_conn.execute.return_value.fetchall.return_value = [
            ("https://example.com", "Example", chrome_time, 5000000),
        ]
        mock_sqlite.connect.return_value = mock_conn

        reader = ChromeReader()
        with patch.object(reader, "_db_path") as mock_db:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_db.return_value = mock_path

            entries = reader.read(since=0)

        assert len(entries) == 1
        assert entries[0].url == "https://example.com"
        assert entries[0].title == "Example"
        assert entries[0].browser == "chrome"
        assert abs(entries[0].visit_time - 1700000000) < 1
        assert entries[0].duration == 5.0

    def test_read_missing_db(self):
        reader = ChromeReader()
        with patch.object(reader, "_db_path") as mock_db:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_db.return_value = mock_path
            entries = reader.read(since=0)
        assert entries == []

    def test_parse_row_no_duration(self):
        reader = ChromeReader()
        chrome_time = (1700000000 + 11644473600) * 1_000_000
        entry = reader._parse_row(("https://x.com", "X", chrome_time, 0))
        assert entry.duration is None

    def test_parse_row_no_title(self):
        reader = ChromeReader()
        chrome_time = (1700000000 + 11644473600) * 1_000_000
        entry = reader._parse_row(("https://x.com", None, chrome_time, 0))
        assert entry.title == ""

    def test_db_path_uses_localappdata(self):
        with patch.dict("os.environ", {"LOCALAPPDATA": "C:\\Users\\test\\AppData\\Local"}):
            reader = ChromeReader()
            path = reader._db_path()
            assert "Google" in str(path)
            assert "Chrome" in str(path)

    @patch("homie_core.browser.readers.sqlite3")
    @patch("homie_core.browser.readers.shutil")
    def test_read_db_error_returns_empty(self, mock_shutil, mock_sqlite):
        mock_sqlite.connect.side_effect = Exception("DB locked")

        reader = ChromeReader()
        with patch.object(reader, "_db_path") as mock_db:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_db.return_value = mock_path
            entries = reader.read(since=0)
        assert entries == []


class TestFirefoxReader:
    def test_db_path_no_profiles(self):
        with patch.dict("os.environ", {"APPDATA": "/tmp/fake"}):
            reader = FirefoxReader()
            path = reader._db_path()
            assert str(path) == "nonexistent"

    def test_parse_row(self):
        reader = FirefoxReader()
        # Firefox: microseconds since Unix epoch
        ff_time = 1700000000 * 1_000_000
        entry = reader._parse_row(("https://mozilla.org", "Mozilla", ff_time))
        assert entry.url == "https://mozilla.org"
        assert entry.browser == "firefox"
        assert abs(entry.visit_time - 1700000000) < 1

    def test_parse_row_no_title(self):
        reader = FirefoxReader()
        entry = reader._parse_row(("https://mozilla.org", None, 0))
        assert entry.title == ""

    @patch("homie_core.browser.readers.sqlite3")
    @patch("homie_core.browser.readers.shutil")
    def test_read_success(self, mock_shutil, mock_sqlite):
        mock_conn = MagicMock()
        ff_time = 1700000000 * 1_000_000
        mock_conn.execute.return_value.fetchall.return_value = [
            ("https://mozilla.org", "Mozilla", ff_time),
        ]
        mock_sqlite.connect.return_value = mock_conn

        reader = FirefoxReader()
        with patch.object(reader, "_db_path") as mock_db:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_db.return_value = mock_path
            entries = reader.read(since=0)

        assert len(entries) == 1
        assert entries[0].browser == "firefox"


class TestEdgeReader:
    def test_db_path_uses_localappdata(self):
        with patch.dict("os.environ", {"LOCALAPPDATA": "C:\\Users\\test\\AppData\\Local"}):
            reader = EdgeReader()
            path = reader._db_path()
            assert "Edge" in str(path)

    def test_parse_row_sets_edge_browser(self):
        reader = EdgeReader()
        chrome_time = (1700000000 + 11644473600) * 1_000_000
        entry = reader._parse_row(("https://bing.com", "Bing", chrome_time, 0))
        assert entry.browser == "edge"

    def test_query_matches_chrome(self):
        edge_reader = EdgeReader()
        chrome_reader = ChromeReader()
        assert edge_reader._query() == chrome_reader._query()

    @patch("homie_core.browser.readers.sqlite3")
    @patch("homie_core.browser.readers.shutil")
    def test_read_success(self, mock_shutil, mock_sqlite):
        mock_conn = MagicMock()
        chrome_time = (1700000000 + 11644473600) * 1_000_000
        mock_conn.execute.return_value.fetchall.return_value = [
            ("https://bing.com", "Bing", chrome_time, 0),
        ]
        mock_sqlite.connect.return_value = mock_conn

        reader = EdgeReader()
        with patch.object(reader, "_db_path") as mock_db:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_db.return_value = mock_path
            entries = reader.read(since=0)

        assert len(entries) == 1
        assert entries[0].browser == "edge"
