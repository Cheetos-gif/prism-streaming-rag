"""Tests for FastAPI endpoints in controller/main.py."""

import pytest
from fastapi.testclient import TestClient

from controller.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_root_redirects_to_dashboard(client):
    res = client.get("/", follow_redirects=False)
    assert res.status_code in (302, 307)
    # The UI is mounted at /dashboard/, so redirecting straight there avoids the
    # extra 307 the StaticFiles mount would issue for /dashboard.
    assert res.headers["location"] == "/dashboard/"


def test_scenarios_endpoint(client):
    res = client.get("/api/scenarios")
    assert res.status_code == 200
    data = res.json()
    assert "scenarios" in data
    assert len(data["scenarios"]) >= 3
    ids = [s["id"] for s in data["scenarios"]]
    assert "field_service" in ids
    assert "workshop" in ids


def test_session_lifecycle(client):
    # Create session
    create_res = client.post("/session")
    assert create_res.status_code == 200
    sid = create_res.json()["session_id"]
    assert sid

    # Get session details
    get_res = client.get(f"/session/{sid}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["session_id"] == sid
    assert data["version"] == 0


def test_corpus_summary_endpoint(client):
    res = client.get("/api/corpus")
    assert res.status_code == 200
    data = res.json()
    assert "total_documents" in data
    assert data["total_documents"] >= 10
    assert "documents" in data


def test_eval_endpoint(client):
    res = client.post("/api/eval")
    assert res.status_code == 200
    data = res.json()
    assert "gates" in data
    assert len(data["gates"]) == 5
    assert data["all_passed"] is True


def test_direct_search_endpoint(client):
    res = client.post("/api/search", json={"query": "Model 7 compressor knocking", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) <= 3
    assert len(data["results"]) > 0
    assert "score" in data["results"][0]
