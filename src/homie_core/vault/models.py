"""Vault data models — dataclasses for credentials, profiles, consent, and finance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Credential:
    id: str
    provider: str
    account_id: str
    token_type: str
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None
    scopes: Optional[list[str]] = None
    active: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class UserProfile:
    id: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class ConsentEntry:
    id: int
    provider: str
    action: str
    scopes: Optional[list[str]] = None
    reason: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class FinancialRecord:
    id: int
    source: str
    category: str
    description: str
    amount: Optional[str] = None
    currency: Optional[str] = None
    due_date: Optional[float] = None
    status: str = "pending"
    reminded_at: Optional[float] = None
    raw_extract: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class ConnectionStatus:
    provider: str
    connected: bool = False
    display_label: Optional[str] = None
    connection_mode: str = "always_on"
    sync_interval: int = 300
    last_sync: Optional[float] = None
    last_sync_error: Optional[str] = None
