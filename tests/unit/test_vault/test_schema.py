import sqlite3
import json
import pytest

from homie_core.vault.schema import (
    create_vault_db, create_cache_db,
    get_schema_version, run_migrations,
    check_integrity, CURRENT_SCHEMA_VERSION,
)


class TestCreateVaultDb:
    def test_creates_tables(self, tmp_path):
        db_path = tmp_path / "vault.db"
        conn = create_vault_db(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert "credentials" in tables
        assert "user_profiles" in tables
        assert "consent_log" in tables
        assert "financial_data" in tables
        conn.close()

    def test_wal_mode_enabled(self, tmp_path):
        db_path = tmp_path / "vault.db"
        conn = create_vault_db(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_credentials_columns(self, tmp_path):
        db_path = tmp_path / "vault.db"
        conn = create_vault_db(db_path)
        cursor = conn.execute("PRAGMA table_info(credentials)")
        cols = {row[1] for row in cursor.fetchall()}
        expected = {"id", "provider", "account_id", "token_type", "access_token",
                    "refresh_token", "expires_at", "scopes", "active",
                    "created_at", "updated_at"}
        assert expected == cols
        conn.close()


class TestCreateCacheDb:
    def test_creates_tables(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert "folder_watches" in tables
        assert "content_index" in tables
        assert "connection_status" in tables
        conn.close()

    def test_connection_status_columns(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        cursor = conn.execute("PRAGMA table_info(connection_status)")
        cols = {row[1] for row in cursor.fetchall()}
        expected = {"provider", "connected", "display_label", "connection_mode",
                    "sync_interval", "last_sync", "last_sync_error"}
        assert expected == cols
        conn.close()


class TestSchemaVersioning:
    def test_initial_version(self, tmp_path):
        meta_path = tmp_path / "vault.meta.json"
        version = get_schema_version(meta_path)
        assert version == 0

    def test_run_migrations_sets_version(self, tmp_path):
        db_path = tmp_path / "vault.db"
        meta_path = tmp_path / "vault.meta.json"
        conn = create_vault_db(db_path)
        run_migrations(conn, meta_path)
        version = get_schema_version(meta_path)
        assert version == CURRENT_SCHEMA_VERSION
        conn.close()


class TestIntegrityCheck:
    def test_valid_db_passes_check(self, tmp_path):
        db_path = tmp_path / "vault.db"
        conn = create_vault_db(db_path)
        assert check_integrity(conn) is True
        conn.close()
