import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("EQUITYMIND_SERVE_WEB_UI", "false")


@pytest.fixture(scope="module")
def client():
    import server

    with TestClient(server.app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_rejects_bad_ticker(client):
    r = client.post("/v1/analyze", json={"ticker": "not a ticker!"})
    assert r.status_code == 422


def test_memory_empty_user(client):
    r = client.get("/v1/users/nobody/memory")
    assert r.status_code == 200
    assert r.json()["watchlist"] == []


def test_adk_app_listed(client):
    r = client.get("/list-apps")
    assert r.status_code == 200
    assert "equitymind" in r.json()
