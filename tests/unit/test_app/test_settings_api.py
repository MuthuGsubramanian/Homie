"""Tests for settings API endpoint."""
from __future__ import annotations
from unittest.mock import MagicMock, PropertyMock
import pytest
from fastapi.testclient import TestClient
from homie_app.tray.dashboard import create_dashboard_app


@pytest.fixture
def mock_inference():
    router = MagicMock()
    type(router).active_source = PropertyMock(return_value="Local")
    type(router).fallback_banner = PropertyMock(return_value=None)
    return router


@pytest.fixture
def mock_email():
    svc = MagicMock()
    svc.get_summary.return_value = {"total": 10, "unread": 3, "high_priority": []}
    svc._providers = {"user@gmail.com": MagicMock()}
    return svc


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.user_name = "Muthu"
    cfg.inference.priority = ["local", "lan", "qubrid"]
    cfg.inference.qubrid.enabled = False
    cfg.inference.vertex.enabled = False
    cfg.llm.model_path = "glm-4.7-flash"
    cfg.llm.backend = "gguf"
    return cfg


@pytest.fixture
def client(mock_inference, mock_email, mock_config):
    app = create_dashboard_app(
        config=mock_config,
        email_service=mock_email,
        inference_router=mock_inference,
        session_token="test-token",
    )
    return TestClient(app)


def test_settings_returns_status(client):
    resp = client.get("/api/settings", cookies={"homie_session": "test-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert "inference" in data
    assert "email" in data
    assert "privacy" in data


def test_settings_inference_info(client):
    resp = client.get("/api/settings", cookies={"homie_session": "test-token"})
    data = resp.json()
    assert data["inference"]["active_source"] == "Local"
    assert data["inference"]["priority"] == ["local", "lan", "qubrid"]


def test_settings_email_accounts(client):
    resp = client.get("/api/settings", cookies={"homie_session": "test-token"})
    data = resp.json()
    assert len(data["email"]["accounts"]) == 1
    assert "user@gmail.com" in data["email"]["accounts"]


def test_settings_requires_auth(client):
    resp = client.get("/api/settings")
    assert resp.status_code == 401


def test_settings_no_services():
    app = create_dashboard_app(session_token="tok")
    c = TestClient(app)
    resp = c.get("/api/settings", cookies={"homie_session": "tok"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["inference"]["active_source"] == "Not configured"
    assert data["email"]["accounts"] == []


def test_settings_page_returns_html(client):
    resp = client.get("/settings", cookies={"homie_session": "test-token"})
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Settings" in resp.text
