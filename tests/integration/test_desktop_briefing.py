"""Integration test: dashboard serves briefing with mocked email service."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from homie_app.tray.dashboard import create_dashboard_app


@pytest.fixture
def email_service():
    svc = MagicMock()
    svc.get_summary.return_value = {
        "total": 10, "unread": 3,
        "high_priority": [{"subject": "Deploy alert", "sender": "ci@ops.io"}],
    }
    svc.get_unread.return_value = {
        "high": [{"id": "h1", "subject": "Deploy alert", "sender": "ci@ops.io", "snippet": "Build failed"}],
        "medium": [{"id": "m1", "subject": "Standup notes", "sender": "team@co.io", "snippet": "Yesterday..."}],
        "low": [],
    }
    svc.get_intelligent_digest.return_value = "3 unread emails. 1 deploy alert needs attention."
    svc.triage.return_value = {"status": "Triaged 3 emails", "action_needed": [], "important": [], "likely_spam": [], "all": []}
    return svc


def test_full_briefing_flow(email_service):
    """Test: server starts, briefing loads, APIs respond, mark-read works."""
    token = "integration-test-token"
    app = create_dashboard_app(email_service=email_service, session_token=token)
    client = TestClient(app)
    cookies = {"homie_session": token}

    # Health (no auth)
    assert client.get("/api/health").status_code == 200

    # Briefing page
    resp = client.get("/briefing", cookies=cookies)
    assert resp.status_code == 200
    assert "Deploy alert" in resp.text
    assert "ci@ops.io" in resp.text

    # Email summary API
    resp = client.get("/api/email/summary", cookies=cookies)
    assert resp.json()["unread"] == 3

    # Triage API
    resp = client.post("/api/email/triage", cookies=cookies)
    assert resp.json()["status"] == "Triaged 3 emails"

    # Mark read
    resp = client.post("/api/email/mark-read/h1", cookies=cookies)
    assert resp.status_code == 200
    email_service.mark_read.assert_called_with("h1")

    # Unauthorized access blocked
    assert client.get("/api/email/summary").status_code == 401
    assert client.get("/briefing").status_code == 401


def test_briefing_email_unread_endpoint(email_service):
    """Test: /api/email/unread returns high/medium/low categorized emails."""
    token = "integration-test-token"
    app = create_dashboard_app(email_service=email_service, session_token=token)
    client = TestClient(app)
    cookies = {"homie_session": token}

    resp = client.get("/api/email/unread", cookies=cookies)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["high"]) == 1
    assert data["high"][0]["subject"] == "Deploy alert"
    assert len(data["medium"]) == 1
    assert data["medium"][0]["subject"] == "Standup notes"
    assert len(data["low"]) == 0


def test_briefing_digest_endpoint(email_service):
    """Test: /api/email/digest returns AI-generated summary."""
    token = "integration-test-token"
    app = create_dashboard_app(email_service=email_service, session_token=token)
    client = TestClient(app)
    cookies = {"homie_session": token}

    resp = client.get("/api/email/digest", cookies=cookies)
    assert resp.status_code == 200
    data = resp.json()
    assert "digest" in data
    assert "deploy alert" in data["digest"].lower()


def test_briefing_no_email_service():
    """Test: dashboard works without email service (empty states)."""
    token = "integration-test-token"
    app = create_dashboard_app(email_service=None, session_token=token)
    client = TestClient(app)
    cookies = {"homie_session": token}

    # Summary returns defaults
    resp = client.get("/api/email/summary", cookies=cookies)
    data = resp.json()
    assert data["total"] == 0
    assert data["unread"] == 0
    assert data["high_priority"] == []

    # Unread returns empty
    resp = client.get("/api/email/unread", cookies=cookies)
    data = resp.json()
    assert data["high"] == []
    assert data["medium"] == []
    assert data["low"] == []

    # Digest returns not-configured message
    resp = client.get("/api/email/digest", cookies=cookies)
    data = resp.json()
    assert "not configured" in data["digest"].lower()

    # Briefing page still renders (with empty state)
    resp = client.get("/briefing", cookies=cookies)
    assert resp.status_code == 200
    assert "Email not configured" in resp.text or "inbox zero" in resp.text.lower()


def test_session_auth_enforced(email_service):
    """Test: all protected endpoints reject requests without valid session."""
    token = "integration-test-token"
    app = create_dashboard_app(email_service=email_service, session_token=token)
    client = TestClient(app)

    # No cookie at all
    assert client.get("/api/email/summary").status_code == 401
    assert client.get("/api/email/unread").status_code == 401
    assert client.post("/api/email/triage").status_code == 401
    assert client.get("/api/email/digest").status_code == 401
    assert client.get("/briefing").status_code == 401
    assert client.post("/api/email/mark-read/h1").status_code == 401

    # Wrong token
    bad_cookies = {"homie_session": "wrong-token"}
    assert client.get("/api/email/summary", cookies=bad_cookies).status_code == 401
    assert client.get("/briefing", cookies=bad_cookies).status_code == 401

    # Health endpoint still works (no auth)
    assert client.get("/api/health").status_code == 200


def test_briefing_page_html_structure(email_service):
    """Test: briefing page HTML includes email details and interactive elements."""
    token = "integration-test-token"
    app = create_dashboard_app(email_service=email_service, session_token=token)
    client = TestClient(app)
    cookies = {"homie_session": token}

    resp = client.get("/briefing", cookies=cookies)
    assert resp.status_code == 200
    html = resp.text

    # Check structure
    assert "<!DOCTYPE html>" in html
    assert "Morning Briefing" in html or "briefing" in html.lower()
    assert "Homie AI" in html

    # Check email content
    assert "Deploy alert" in html
    assert "ci@ops.io" in html
    assert "Build failed" in html
    assert "Standup notes" in html
    assert "team@co.io" in html
    assert "Yesterday..." in html

    # Check stats
    assert "Unread" in html
    assert "3" in html  # 3 unread
    assert "10" in html  # 10 total

    # Check AI summary
    assert "3 unread emails" in html
    assert "deploy alert" in html.lower()

    # Check interactive buttons
    assert "Mark read" in html or "mark" in html.lower()
