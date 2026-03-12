"""Homie Secure Vault — encrypted local storage for credentials and user data."""

from homie_core.vault.exceptions import (
    VaultError,
    VaultLockedError,
    VaultAuthError,
    VaultCorruptError,
    CredentialNotFoundError,
    RateLimitError,
)

__all__ = [
    "VaultError",
    "VaultLockedError",
    "VaultAuthError",
    "VaultCorruptError",
    "CredentialNotFoundError",
    "RateLimitError",
]
