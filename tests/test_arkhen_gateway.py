import pytest
from fastapi.testclient import TestClient
from arkhen.api.main import app, CodeNode

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ARKHE(N) GATEWAY ONLINE"

def test_code_metrics():
    node_data = {
        "id": "test-node",
        "lines_of_code": 100,
        "cyclomatic_complexity": 5.0,
        "test_coverage": 0.8,
        "coupling_to_others": ["node-a", "node-b"],
        "user_satisfaction": 0.9,
        "commits_since_refactor": 10,
        "bugs_critical": 0
    }
    response = client.post("/code/metrics", json=node_data)
    assert response.status_code == 200
    data = response.json()
    assert "vk" in data
    assert "q" in data
    assert data["q"] > 0.5

def test_synchronize_array():
    array_data = {
        "n_emitters": 10,
        "coupling_strength": 0.5,
        "natural_frequencies": [0.1] * 10,
        "phases": [0.0] * 10
    }
    response = client.post("/array/synchronize", json=array_data)
    assert response.status_code == 200
    data = response.json()
    assert "coherence" in data
    assert "permeability" in data
    assert 0 <= data["coherence"] <= 1.0

def test_kuramoto_logic():
    # Test internal logic
    node = CodeNode(
        id="core",
        lines_of_code=500,
        cyclomatic_complexity=10,
        test_coverage=1.0,
        coupling_to_others=[],
        user_satisfaction=1.0,
        commits_since_refactor=1,
        bugs_critical=0
    )
    vk = node.compute_vk()
    assert vk["bio"] == 1.0
    assert vk["cog"] == 0.5
    assert node.compute_q() > 0.0
