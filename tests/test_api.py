from fastapi.testclient import TestClient
import sys
import os

# Ensure gateway is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway.app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["identity"] == "Arkhe(n) DMR Service"

def test_calibrate():
    response = client.post(
        "/agent/test-agent/calibrate",
        json={"bio": 0.5, "aff": 0.5, "soc": 0.5, "cog": 0.5}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "calibrated"

def test_vk_trajectory():
    response = client.get("/agent/test-agent/vk")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
