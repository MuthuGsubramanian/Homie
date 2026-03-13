import pytest
from homie_core.vault.exceptions import (
    VaultError, VaultLockedError, VaultAuthError, VaultCorruptError,
    CredentialNotFoundError, RateLimitError,
)

class TestVaultExceptionHierarchy:
    def test_all_inherit_from_vault_error(self):
        for exc_cls in (VaultLockedError, VaultAuthError, VaultCorruptError,
                        CredentialNotFoundError, RateLimitError):
            assert issubclass(exc_cls, VaultError)

    def test_vault_error_is_exception(self):
        assert issubclass(VaultError, Exception)

    def test_exceptions_carry_message(self):
        err = VaultLockedError("vault is locked")
        assert str(err) == "vault is locked"

    def test_rate_limit_carries_retry_after(self):
        err = RateLimitError("too many attempts", retry_after=60.0)
        assert err.retry_after == 60.0
