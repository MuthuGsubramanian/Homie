"""Vault exception hierarchy."""
from __future__ import annotations


class VaultError(Exception):
    """Base exception for all vault operations."""


class VaultLockedError(VaultError):
    """Raised when an operation is attempted on a locked vault."""


class VaultAuthError(VaultError):
    """Raised on wrong password or corrupted master key."""


class VaultCorruptError(VaultError):
    """Raised on database corruption or tampered ciphertext (GCM auth fail)."""


class CredentialNotFoundError(VaultError):
    """Raised when a requested credential does not exist."""


class RateLimitError(VaultError):
    """Raised after too many failed unlock attempts."""

    def __init__(self, message: str = "", retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after
