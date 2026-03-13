from __future__ import annotations

import io
import tarfile
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64
import os


class BackupManager:
    SALT_SIZE = 16
    ITERATIONS = 480_000

    def _derive_key(self, passphrase: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))

    def create_backup(self, source: Path, dest: Path, passphrase: str) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(source), arcname=".")
        plaintext = buf.getvalue()
        salt = os.urandom(self.SALT_SIZE)
        key = self._derive_key(passphrase, salt)
        f = Fernet(key)
        ciphertext = f.encrypt(plaintext)
        out_path = dest / "homie_backup.enc"
        out_path.write_bytes(salt + ciphertext)
        return out_path

    def restore_backup(self, backup_dir: Path, dest: Path, passphrase: str) -> None:
        enc_path = backup_dir / "homie_backup.enc"
        raw = enc_path.read_bytes()
        salt = raw[: self.SALT_SIZE]
        ciphertext = raw[self.SALT_SIZE :]
        key = self._derive_key(passphrase, salt)
        f = Fernet(key)
        try:
            plaintext = f.decrypt(ciphertext)
        except InvalidToken:
            raise ValueError("Wrong passphrase or corrupted backup")
        dest.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO(plaintext)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(str(dest), filter="data")
