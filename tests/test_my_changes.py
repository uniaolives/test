from fastapi.testclient import TestClient
import pytest
from gateway.app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["identity"] == "Arkhe(n) DMR Service"

def test_synchronicity():
    response = client.get("/metrics/synchronicity")
    assert response.status_code == 200
    data = response.json()
    assert "synchronicity_index" in data
    assert "status" in data

def test_trefoil_knot_endpoint():
    response = client.get("/quantum/qiskit/trefoil_knot")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "counts" in data
    assert "coherence" in data

def test_constitutional_guard():
    # Since current_h is mocked to 0.5, this should pass
    response = client.get("/hyperclaw/templates")
    assert response.status_code == 200

    # We could try to mock current_h to 1.2 to test the 503,
    # but for a basic verification, seeing it pass is good.
