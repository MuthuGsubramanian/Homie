# Privacy & Secure Storage Layer — Design Spec

**Date:** 2026-03-12
**Sub-project:** #5 of 5 (Foundation layer)
**Branch:** feat/homie-ai-v2
**Dependencies:** None (other sub-projects depend on this)

---

## Overview

A local-first, encrypted storage layer that manages credentials, user profiles, consent records, and financial data for Homie's integrations (email, social media, local folders). All sensitive data stays on the user's machine, encrypted at rest with AES-256-GCM, with the master key protected by the OS keyring.

This module is the foundation for all integration sub-projects:
- Sub-project 1: Email Integration
- Sub-project 2: Financial Intelligence
- Sub-project 3: Local Folder Awareness
- Sub-project 4: Social & Communication Connectors

---

## Encryption Architecture

### Key Hierarchy

```
OS Keyring (Windows Credential Manager / macOS Keychain / Linux Secret Service)
  └── master_key (32 bytes, os.urandom)
      └── Optional password layer: PBKDF2(password, salt, 600k iterations)
          └── HKDF-SHA256(master_key, salt=category_name) per category:
              ├── credentials_key
              ├── profiles_key
              ├── financial_key
              └── consent_key
```

### Encryption Details

- **Algorithm:** AES-256-GCM (authenticated encryption with tamper detection)
- **Per-field encryption:** Each sensitive field encrypted independently with its own random 12-byte nonce
- **Storage format:** `nonce (12B) || ciphertext || tag (16B)`, base64-encoded in SQLite TEXT columns
- **Key derivation:** HKDF-SHA256 from master key + category name as salt
- **Password layer:** Optional. PBKDF2 with 600k iterations encrypts the master key before keyring storage
- **Key material:** Stored in mutable `bytearray` (not `bytes`) so keys can be zeroed on lock

---

## Database Schema

Two SQLite databases with clear separation of encrypted and plaintext data.

### vault.db (encrypted fields, file permissions 0600)

```sql
-- OAuth tokens, API keys, refresh tokens
CREATE TABLE credentials (
    id TEXT PRIMARY KEY,           -- "gmail:user@example.com"
    provider TEXT NOT NULL,        -- "gmail", "linkedin", "slack", etc.
    token_type TEXT NOT NULL,      -- "oauth2", "api_key", "app_password"
    access_token TEXT NOT NULL,    -- AES-256-GCM encrypted
    refresh_token TEXT,            -- AES-256-GCM encrypted
    expires_at REAL,               -- Unix timestamp (plaintext for query efficiency)
    scopes TEXT,                   -- JSON array of granted scopes (plaintext)
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- User identity and linked account metadata
CREATE TABLE user_profiles (
    id TEXT PRIMARY KEY,           -- "primary" or provider-specific
    display_name TEXT,             -- AES-256-GCM encrypted
    email TEXT,                    -- AES-256-GCM encrypted
    phone TEXT,                    -- AES-256-GCM encrypted
    metadata TEXT,                 -- AES-256-GCM encrypted JSON
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Full audit trail of user consent decisions (append-only)
CREATE TABLE consent_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,        -- "gmail", "linkedin", etc.
    action TEXT NOT NULL,          -- "connected", "declined", "disconnected", "reauthorized"
    scopes TEXT,                   -- What was granted/revoked
    reason TEXT,                   -- Why (user-initiated, just-in-time prompt, etc.)
    timestamp REAL NOT NULL
);

-- Bills, payments, due dates extracted from emails
CREATE TABLE financial_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- "gmail:msg_id_123"
    category TEXT NOT NULL,        -- "bill", "payment", "subscription", "reminder"
    description TEXT NOT NULL,     -- AES-256-GCM encrypted
    amount TEXT,                   -- AES-256-GCM encrypted (string to avoid float issues)
    currency TEXT,                 -- "USD", "INR", etc. (plaintext)
    due_date REAL,                 -- Unix timestamp (plaintext for query efficiency)
    status TEXT DEFAULT 'pending', -- "pending", "paid", "overdue", "dismissed"
    reminded_at REAL,              -- Last reminder timestamp
    raw_extract TEXT,              -- AES-256-GCM encrypted (original email snippet)
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
```

### cache.db (plaintext, file permissions 0644)

```sql
-- Folder watch configuration
CREATE TABLE folder_watches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    label TEXT,
    scan_interval INTEGER DEFAULT 300,
    last_scanned REAL,
    file_count INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1
);

-- Content analysis results and indexes
CREATE TABLE content_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- "folder:/path/file" or "email:msg_id"
    content_type TEXT NOT NULL,    -- "document", "code", "image", "spreadsheet"
    summary TEXT,
    topics TEXT,                   -- JSON array of detected topics
    embeddings BLOB,              -- Vector embeddings for RAG
    indexed_at REAL NOT NULL
);

-- Connection status (fast lookup without decrypting vault)
CREATE TABLE connection_status (
    provider TEXT PRIMARY KEY,
    connected INTEGER DEFAULT 0,
    display_label TEXT,            -- "Gmail (m***@gmail.com)" - partially masked
    connection_mode TEXT DEFAULT 'always_on',  -- "always_on" or "on_demand"
    sync_interval INTEGER DEFAULT 300,         -- seconds, per provider
    last_sync REAL,
    last_sync_error TEXT
);
```

---

## SecureVault API

```python
class SecureVault:
    """Single entry point for all secure storage operations."""

    def __init__(self, storage_dir: str | Path = "~/.homie/vault"): ...

    # Lifecycle
    def unlock(self, password: str | None = None) -> None: ...
    def lock(self) -> None:                     # Zeros all derived keys in memory
    def is_unlocked(self) -> bool: ...
    def set_password(self, password: str) -> None: ...
    def export_vault(self, path: Path, password: str) -> None: ...
    def import_vault(self, path: Path, password: str) -> None: ...

    # Credentials
    def store_credential(self, provider: str, token_type: str,
                        access_token: str, refresh_token: str | None = None,
                        expires_at: float | None = None,
                        scopes: list[str] | None = None) -> None: ...
    def get_credential(self, provider: str) -> Credential | None: ...
    def refresh_credential(self, provider: str,
                          new_access_token: str,
                          new_expires_at: float | None = None) -> None: ...
    def deactivate_credential(self, provider: str) -> None: ...
    def delete_credential(self, provider: str) -> None: ...

    # User Profiles
    def store_profile(self, provider: str, display_name: str | None = None,
                     email: str | None = None, phone: str | None = None,
                     metadata: dict | None = None) -> None: ...
    def get_profile(self, provider: str) -> UserProfile | None: ...

    # Consent Management
    def log_consent(self, provider: str, action: str,
                   scopes: list[str] | None = None,
                   reason: str | None = None) -> None: ...
    def get_consent_history(self, provider: str) -> list[ConsentEntry]: ...
    def get_last_consent(self, provider: str) -> ConsentEntry | None: ...

    # Financial Data
    def store_financial(self, source: str, category: str,
                       description: str, amount: str | None = None,
                       currency: str | None = None,
                       due_date: float | None = None) -> int: ...
    def query_financial(self, status: str | None = None,
                       due_before: float | None = None,
                       category: str | None = None) -> list[FinancialRecord]: ...
    def update_financial(self, record_id: int, **kwargs) -> None: ...

    # Connection Status
    def set_connection_status(self, provider: str, connected: bool,
                             label: str | None = None,
                             mode: str = "always_on",
                             sync_interval: int = 300) -> None: ...
    def get_connection_status(self, provider: str) -> ConnectionStatus | None: ...
    def get_all_connections(self) -> list[ConnectionStatus]: ...
```

---

## Consent & Connection Flow

### First-time connection (just-in-time prompt)

When a feature needs a provider that isn't connected:

1. Homie explains what access it needs and why, in plain language
2. User approves or declines
3. If approved: OAuth2 browser flow, credentials stored, consent logged
4. User is asked about connection mode:
   - **Always connected (default):** Periodic background sync at configurable intervals
   - **Auto-disconnect (on_demand):** Connect only when needed, disconnect after sync
5. If declined: consent logged as "declined", Homie respects it with a 7-day re-ask cooldown

### Default behavior: always connected

When `connection_mode = "always_on"`:
- Background sync at provider-specific intervals (email: 5min, social: 30min, finance: 6hr)
- Token auto-refresh on expiry (silent, notify on failure)
- Data retrieved and stored locally continuously

When `connection_mode = "on_demand"`:
- Connect only when Homie needs data or user asks
- Disconnect after sync completes
- No background polling

### Disconnection flow

User-initiated disconnect presents three options:
1. **Disconnect (keep credentials encrypted)** — can reconnect later without re-auth
2. **Disconnect and delete credentials permanently** — full removal after confirmation
3. **Cancel**

Credentials are never deleted without explicit user confirmation.

### CLI commands

```
homie connect [provider]              # OAuth flow + consent
homie connect [provider] --mode X     # Set always_on or on_demand
homie connect [provider] --sync-interval N  # Seconds between syncs
homie disconnect [provider]           # Deactivate with confirmation dialog
homie connections                     # List all connection statuses
homie consent-log [provider]          # Full audit trail
```

---

## Master Key Lifecycle

### First run
1. Generate 32-byte random key via `os.urandom(32)`
2. Store in OS keyring (service: `homie-vault`, username: `master-key`)
3. Create `vault.db` and `cache.db` with schema
4. Derive category keys via HKDF
5. Optionally ask user to set a master password

### Session start (daemon boot)
1. Retrieve master key from OS keyring
2. If password-protected: prompt for password, PBKDF2 derive, decrypt
3. Derive category keys via HKDF, hold in memory as `bytearray`
4. Start SyncManager for always-on providers

### Session end / lock
1. Zero all derived keys in memory (overwrite `bytearray` with `\x00`)
2. Close DB connections
3. For on_demand providers: revoke active tokens

---

## Security Hardening

| Threat | Mitigation |
|--------|-----------|
| Memory dump | Zero keys on lock; keys in mutable `bytearray` not `bytes` |
| DB file stolen | Per-field AES-256-GCM; useless without master key from OS keyring |
| Keyring compromised | Optional password layer keeps master key encrypted |
| Token leak in logs | Existing `redact_sensitive_text()` catches OAuth token patterns |
| Prompt injection via email | Existing `injection_detector.py` scans external content |
| Brute force password | PBKDF2 600k iterations; rate limit: 5 failures → 60s cooldown |
| Stale tokens | Auto-refresh with failure notification; never store plaintext passwords |
| Backup exposure | `export_vault()` always password-encrypted, independent of keyring |

### File permissions

```
~/.homie/vault/
├── vault.db          # 0600 (owner read/write only)
├── cache.db          # 0644 (readable, no secrets)
└── vault.meta.json   # 0600 (schema version, created_at)
```

---

## Module Structure

```
src/homie_core/vault/
├── __init__.py
├── secure_vault.py       # SecureVault class — main API
├── encryption.py         # AES-256-GCM, HKDF, PBKDF2 key derivation
├── keyring_backend.py    # OS keyring read/write, password layer
├── schema.py             # DB table creation, migrations
├── models.py             # Credential, UserProfile, ConsentEntry,
│                         #   FinancialRecord, ConnectionStatus dataclasses
└── sync_manager.py       # Periodic background sync (always_on / on_demand)
```

### Integration with Homie

```python
# In HomieDaemon.__init__():
self._vault = SecureVault(storage_dir=storage / "vault")
self._vault.unlock()

# In HomieDaemon.start():
self._sync_manager = SyncManager(vault=self._vault)
self._sync_manager.start()

# In HomieDaemon.stop():
self._sync_manager.stop()
self._vault.lock()
```

SyncManager reuses the daemon's existing 60-second tick pattern rather than spawning a new thread.

### Dependencies

| Package | Purpose | Notes |
|---------|---------|-------|
| `cryptography` | AES-256-GCM, HKDF, PBKDF2 | Well-maintained, widely used |
| `keyring` | OS keyring abstraction | ~50KB, pure Python + platform backends |

SQLite is in Python's stdlib. No other new dependencies.

---

## Build Order (Full Project)

This module (#5) is built first. Subsequent sub-projects in order:

1. **Email Integration (#1)** — Gmail/IMAP, uses vault for OAuth tokens
2. **Financial Intelligence (#2)** — builds on email data, uses vault for financial records
3. **Local Folder Awareness (#3)** — uses cache.db for folder config and indexes
4. **Social & Communication Connectors (#4)** — uses vault for social OAuth tokens and consent
