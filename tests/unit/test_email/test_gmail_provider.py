"""Tests for Gmail provider — all API calls mocked."""
from __future__ import annotations

import base64
import time
from unittest.mock import MagicMock, patch, PropertyMock

from homie_core.email.gmail_provider import GmailProvider
from homie_core.email.models import EmailMessage, Label


def _mock_service():
    """Create a mock Gmail API service object."""
    return MagicMock()


def _make_gmail_message(msg_id="msg1", thread_id="t1", subject="Test",
                         sender="alice@x.com", to="user@gmail.com",
                         snippet="Hello...", date_ms=1710288000000,
                         labels=None, has_body=True):
    """Build a Gmail API message response dict."""
    headers = [
        {"name": "Subject", "value": subject},
        {"name": "From", "value": sender},
        {"name": "To", "value": to},
        {"name": "Date", "value": "Tue, 12 Mar 2024 12:00:00 +0000"},
    ]
    msg = {
        "id": msg_id,
        "threadId": thread_id,
        "snippet": snippet,
        "labelIds": labels or ["INBOX"],
        "internalDate": str(date_ms),
        "payload": {
            "headers": headers,
            "mimeType": "text/plain",
        },
    }
    if has_body:
        msg["payload"]["body"] = {
            "data": base64.urlsafe_b64encode(b"Hello body").decode()
        }
    return msg


class TestGmailProviderParsing:
    def test_parse_message(self):
        provider = GmailProvider(account_id="user@gmail.com")
        raw = _make_gmail_message()
        msg = provider._parse_message(raw)
        assert msg.id == "msg1"
        assert msg.subject == "Test"
        assert msg.sender == "alice@x.com"
        assert msg.provider == "gmail"

    def test_parse_message_with_attachments(self):
        provider = GmailProvider(account_id="user@gmail.com")
        raw = _make_gmail_message()
        raw["payload"]["parts"] = [
            {"filename": "doc.pdf", "mimeType": "application/pdf", "body": {"size": 1024}},
        ]
        msg = provider._parse_message(raw)
        assert msg.has_attachments is True
        assert "doc.pdf" in msg.attachment_names


class TestGmailProviderFetch:
    def test_fetch_messages(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        # Mock messages().list()
        service.users().messages().list().execute.return_value = {
            "messages": [{"id": "msg1"}, {"id": "msg2"}],
        }
        # Mock messages().get()
        service.users().messages().get().execute.side_effect = [
            _make_gmail_message(msg_id="msg1"),
            _make_gmail_message(msg_id="msg2", subject="Second"),
        ]

        messages = provider.fetch_messages(since=0.0, max_results=10)
        assert len(messages) == 2

    def test_search(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().messages().list().execute.return_value = {
            "messages": [{"id": "msg1"}],
        }
        service.users().messages().get().execute.return_value = _make_gmail_message()

        results = provider.search("from:alice")
        assert len(results) == 1


class TestGmailProviderActions:
    def test_apply_label(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        provider.apply_label("msg1", "Label_1")
        service.users().messages().modify.assert_called()

    def test_trash(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        provider.trash("msg1")
        service.users().messages().trash.assert_called()

    def test_create_draft(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().drafts().create().execute.return_value = {"id": "draft1"}

        draft_id = provider.create_draft(to="bob@x.com", subject="Re: Hi", body="Hello Bob")
        assert draft_id == "draft1"

    def test_get_profile(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().getProfile().execute.return_value = {
            "emailAddress": "user@gmail.com",
        }

        profile = provider.get_profile()
        assert profile["emailAddress"] == "user@gmail.com"

    def test_list_labels(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().labels().list().execute.return_value = {
            "labels": [
                {"id": "INBOX", "name": "INBOX", "type": "system"},
                {"id": "Label_1", "name": "Homie/Bills", "type": "user"},
            ],
        }

        labels = provider.list_labels()
        assert len(labels) == 2
        assert labels[0].id == "INBOX"


from homie_core.email.models import EmailThread


class TestGmailProviderThreads:
    def test_get_thread(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().threads().get().execute.return_value = {
            "id": "t1",
            "messages": [
                _make_gmail_message(msg_id="msg1", thread_id="t1"),
                _make_gmail_message(msg_id="msg2", thread_id="t1", subject="Re: Test"),
            ],
        }

        thread = provider.get_thread("t1")
        assert thread.id == "t1"
        assert len(thread.messages) == 2
        assert thread.messages[0].id == "msg1"

    def test_get_inbox_threads(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().threads().list().execute.return_value = {
            "threads": [{"id": "t1"}, {"id": "t2"}],
        }
        service.users().threads().get().execute.side_effect = [
            {"id": "t1", "messages": [_make_gmail_message(msg_id="msg1", thread_id="t1")]},
            {"id": "t2", "messages": [_make_gmail_message(msg_id="msg2", thread_id="t2")]},
        ]

        threads = provider.get_inbox_threads(max_results=10)
        assert len(threads) == 2

    def test_get_unread_count(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().labels().get().execute.return_value = {"messagesUnread": 5}
        count = provider.get_unread_count("INBOX")
        assert count == 5

    def test_archive_thread(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        provider.archive_thread("t1")
        service.users().threads().modify.assert_called()

    def test_trash_thread(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        provider.trash_thread("t1")
        service.users().threads().trash.assert_called()


from homie_core.email.models import EmailDraft


class TestGmailProviderDrafts:
    def test_list_drafts(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().drafts().list().execute.return_value = {
            "drafts": [{"id": "d1", "message": _make_gmail_message(msg_id="msg1")}],
        }
        service.users().drafts().get().execute.return_value = {
            "id": "d1",
            "message": _make_gmail_message(msg_id="msg1"),
        }
        drafts = provider.list_drafts()
        assert len(drafts) == 1
        assert drafts[0].id == "d1"

    def test_get_draft(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().drafts().get().execute.return_value = {
            "id": "d1",
            "message": _make_gmail_message(msg_id="msg1", subject="My Draft"),
        }
        draft = provider.get_draft("d1")
        assert draft.id == "d1"
        assert draft.message.subject == "My Draft"

    def test_delete_draft(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        provider.delete_draft("d1")
        service.users().drafts().delete.assert_called()

    def test_update_draft(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().drafts().update().execute.return_value = {"id": "d1"}
        result = provider.update_draft("d1", to="bob@x.com", subject="Updated", body="New body")
        assert result == "d1"
