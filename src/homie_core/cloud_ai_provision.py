"""Optional provisioning of a Qubrid cloud AI key.

Homie ships no cloud key: it is fully local by default. A user who wants the Qubrid
cloud fallback provides their own key once, which is then kept in the encrypted vault:
    HOMIE_QUBRID_KEY=... python -m homie_core.cloud_ai_provision
"""
from __future__ import annotations

import os

from homie_core.vault.secure_vault import SecureVault
from homie_core.cloud_ai import store_cloud_key, is_configured




def provision(vault: SecureVault | None = None) -> bool:
    """Store the user's own Qubrid key (HOMIE_QUBRID_KEY) in the vault. Returns True if newly stored."""
    key = os.environ.get("HOMIE_QUBRID_KEY", "").strip()
    if not key:
        return False
    own_vault = vault is None
    if own_vault:
        vault = SecureVault()
        vault.unlock()
    try:
        if is_configured(vault):
            return False
        store_cloud_key(vault, key)
        return True
    finally:
        if own_vault:
            vault.lock()


if __name__ == "__main__":
    stored = provision()
    if stored:
        print("Qubrid key stored in the vault.")
    else:
        print("Nothing stored: set HOMIE_QUBRID_KEY to your own key, or a key is already configured.")
