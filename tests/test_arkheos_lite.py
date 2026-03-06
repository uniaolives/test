from fastapi.testclient import TestClient
from gateway.app.main import app
import pytest

client = TestClient(app)

def test_arkhe_scheduler_task():
    task = {"id": 1, "coherence": 0.5, "priority": 10}
    response = client.post("/arkhe/scheduler/task", json=task)
    assert response.status_code == 200
    assert response.json()["status"] == "scheduled"
    assert response.json()["phi_q"] > 1.0

def test_arkhe_scheduler_status():
    response = client.get("/arkhe/scheduler/status")
    assert response.status_code == 200
    assert "phi_q" in response.json()
    assert "queue_len" in response.json()

def test_arkhe_ledger_handover():
    handover = {
        "id": 101,
        "source_epoch": 224,
        "target_epoch": 219,
        "coherence": 0.95,
        "payload_hash": "abc"
    }
    response = client.post("/arkhe/ledger/handover", json=handover)
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"

def test_arkhe_ledger_history():
    response = client.get("/arkhe/ledger/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
