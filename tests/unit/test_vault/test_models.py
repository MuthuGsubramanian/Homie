import time
from homie_core.vault.models import (
    Credential, UserProfile, ConsentEntry, FinancialRecord, ConnectionStatus,
)

class TestCredential:
    def test_create_credential(self):
        cred = Credential(
            id="gmail:user@example.com", provider="gmail",
            account_id="user@example.com", token_type="oauth2",
            access_token="access_123", refresh_token="refresh_456",
            expires_at=time.time() + 3600, scopes=["email.read"], active=True,
        )
        assert cred.provider == "gmail"
        assert cred.active is True

    def test_credential_id_format(self):
        cred = Credential(
            id="slack:U12345", provider="slack", account_id="U12345",
            token_type="oauth2", access_token="xoxb-token",
        )
        assert cred.id == "slack:U12345"
        assert cred.refresh_token is None
        assert cred.scopes is None
        assert cred.active is True

class TestUserProfile:
    def test_create_profile(self):
        profile = UserProfile(id="primary", display_name="Muthu", email="muthu@example.com")
        assert profile.display_name == "Muthu"
        assert profile.phone is None
        assert profile.metadata is None

class TestConsentEntry:
    def test_create_consent(self):
        entry = ConsentEntry(id=1, provider="gmail", action="connected",
                            scopes=["email.read"], reason=None, timestamp=time.time())
        assert entry.action == "connected"

class TestFinancialRecord:
    def test_create_financial(self):
        record = FinancialRecord(
            id=1, source="gmail:msg123", category="bill",
            description="Electric bill", amount="142.50", currency="USD",
            due_date=time.time() + 86400, status="pending",
        )
        assert record.category == "bill"
        assert record.reminded_at is None

class TestConnectionStatus:
    def test_create_status(self):
        status = ConnectionStatus(
            provider="gmail", connected=True,
            display_label="Gmail (m***@gmail.com)",
            connection_mode="always_on", sync_interval=300,
        )
        assert status.connected is True
        assert status.connection_mode == "always_on"
