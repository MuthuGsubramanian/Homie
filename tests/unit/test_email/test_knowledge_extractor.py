"""Tests for EmailKnowledgeExtractor."""
from __future__ import annotations

import sqlite3

from homie_core.email.models import EmailMessage


def _make_msg(msg_id="msg1", sender="alice@corp.com", subject="Q1 Review",
              recipients=None, snippet="Please review the Q1 report"):
    return EmailMessage(
        id=msg_id, thread_id="t1", account_id="user@gmail.com",
        provider="gmail", subject=subject, sender=sender,
        recipients=recipients or ["user@gmail.com"],
        snippet=snippet, body=snippet,
    )


class TestHeuristicExtraction:
    def test_extract_contacts(self):
        from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
        conn = sqlite3.connect(":memory:")
        extractor = EmailKnowledgeExtractor(cache_conn=conn, graph=None)
        msg = _make_msg()
        result = extractor._extract_heuristic(msg)
        assert any(e.get("email", "") == "alice@corp.com" for e in result.get("entities", []))

    def test_extract_organization_from_domain(self):
        from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
        conn = sqlite3.connect(":memory:")
        extractor = EmailKnowledgeExtractor(cache_conn=conn, graph=None)
        msg = _make_msg(sender="bob@acmecorp.com")
        result = extractor._extract_heuristic(msg)
        orgs = [e for e in result.get("entities", []) if e.get("type") == "organization"]
        assert len(orgs) >= 1

    def test_extract_deadline_pattern(self):
        from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
        conn = sqlite3.connect(":memory:")
        extractor = EmailKnowledgeExtractor(cache_conn=conn, graph=None)
        msg = _make_msg(snippet="Please submit by Friday March 28")
        result = extractor._extract_heuristic(msg)
        assert len(result.get("action_items", [])) >= 1 or len(result.get("events", [])) >= 1


class TestProcessMessage:
    def test_process_stores_contact(self):
        from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
        conn = sqlite3.connect(":memory:")
        extractor = EmailKnowledgeExtractor(cache_conn=conn, graph=None)
        msg = _make_msg()
        extractor.process_message(msg)
        row = conn.execute("SELECT email FROM email_contacts WHERE email LIKE '%alice%'").fetchone()
        assert row is not None


class TestBatchExtract:
    def test_batch_processes_multiple(self):
        from homie_core.email.knowledge_extractor import EmailKnowledgeExtractor
        conn = sqlite3.connect(":memory:")
        extractor = EmailKnowledgeExtractor(cache_conn=conn, graph=None)
        msgs = [_make_msg(msg_id=f"msg{i}", sender=f"person{i}@x.com") for i in range(5)]
        extractor.batch_extract(msgs)
        count = conn.execute("SELECT COUNT(*) FROM email_contacts").fetchone()[0]
        assert count >= 5
