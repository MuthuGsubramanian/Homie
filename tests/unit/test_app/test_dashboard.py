from fastapi.testclient import TestClient
from homie_app.tray.dashboard import create_dashboard_app


def test_health():
    app = create_dashboard_app()
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_profile_empty():
    app = create_dashboard_app()
    client = TestClient(app)
    resp = client.get("/api/profile")
    assert resp.status_code == 200


def test_beliefs_empty():
    app = create_dashboard_app()
    client = TestClient(app)
    resp = client.get("/api/beliefs")
    assert resp.status_code == 200


def test_plugins_empty():
    app = create_dashboard_app()
    client = TestClient(app)
    resp = client.get("/api/plugins")
    assert resp.status_code == 200
