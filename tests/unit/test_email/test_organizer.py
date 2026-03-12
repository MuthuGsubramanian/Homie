"""Tests for email organizer — labeling, archiving, financial extraction."""
from __future__ import annotations

import re
from unittest.mock import MagicMock, call

from homie_core.email.organizer import EmailOrganizer
from homie_core.email.models import EmailMessage, EmailSyncConfig


def _make_msg(**overrides) -> EmailMessage:
    defaults = dict(
        id="msg1", thread_id="t1", account_id="user@work.com",
        provider="gmail", subject="Hello", sender="alice@example.com",
        recipients=["user@work.com"], snippet="Hey there...",
        priority="medium", spam_score=0.0, categories=[],
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


class TestLabelApplication:
    def test_bill_gets_homie_bills_label(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={"bill": "Label_Bills"})
        msg = _make_msg(categories=["bill"])
        organizer.apply_labels(msg)
        provider.apply_label.assert_called_with("msg1", "Label_Bills")

    def test_multiple_categories_get_multiple_labels(self):
        provider = MagicMock()
        organizer = EmailOrganizer(
            provider=provider,
            label_ids={"bill": "L_B", "work": "L_W"},
        )
        msg = _make_msg(categories=["bill", "work"])
        organizer.apply_labels(msg)
        assert provider.apply_label.call_count == 2

    def test_no_categories_no_labels(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(categories=[])
        organizer.apply_labels(msg)
        provider.apply_label.assert_not_called()


class TestArchiveRules:
    def test_low_priority_not_direct_archived(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(priority="low", recipients=["other@x.com", "user@work.com"])
        result = organizer.should_archive(msg, user_is_direct=False)
        assert result is True

    def test_high_priority_not_archived(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(priority="high")
        result = organizer.should_archive(msg, user_is_direct=True)
        assert result is False

    def test_social_archived(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(categories=["social"])
        result = organizer.should_archive(msg, user_is_direct=True)
        assert result is True

    def test_spam_above_threshold_trashed(self):
        provider = MagicMock()
        config = EmailSyncConfig(account_id="user@work.com", auto_trash_spam=True)
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(spam_score=0.85)
        result = organizer.should_trash(msg, config)
        assert result is True

    def test_spam_below_threshold_not_trashed(self):
        provider = MagicMock()
        config = EmailSyncConfig(account_id="user@work.com", auto_trash_spam=True)
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(spam_score=0.5)
        result = organizer.should_trash(msg, config)
        assert result is False

    def test_auto_trash_disabled(self):
        provider = MagicMock()
        config = EmailSyncConfig(account_id="user@work.com", auto_trash_spam=False)
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(spam_score=0.9)
        result = organizer.should_trash(msg, config)
        assert result is False


class TestFinancialExtraction:
    def test_extract_amount_and_date(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(
            subject="Invoice #4521 — Payment Due March 20",
            snippet="Amount due: $142.50. Please pay by March 20, 2026.",
        )
        result = organizer.extract_financial(msg)
        assert result is not None
        assert result["amount"] == "142.50"
        assert result["currency"] == "USD"

    def test_no_financial_data(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(subject="Hey", snippet="How are you?")
        result = organizer.extract_financial(msg)
        assert result is None
