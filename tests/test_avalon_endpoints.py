
import unittest
from fastapi.testclient import TestClient
from avalon.services.zeitgeist import app as zeitgeist_app
from avalon.services.qhttp_gateway import app as qhttp_app
from avalon.services.starlink_q import app as starlink_app
from avalon.security.f18_safety_guard import safety_check

class TestAvalonEndpoints(unittest.TestCase):
    def test_zeitgeist_metrics(self):
        client = TestClient(zeitgeist_app)
        response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "RESONANT")
        self.assertIn("Z_t", data)
        self.assertIn("h_hausdorff", data)
        # Verify h is safe
        self.assertTrue(1.2 <= data["h_hausdorff"] <= 1.8)

    def test_zeitgeist_calibrate(self):
        client = TestClient(zeitgeist_app)
        # Test damping and safety check
        response = client.post("/calibrate", json={"z_score": 0.95, "h_dim": 1.9})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["active_z"], 0.89) # Clamped
        self.assertEqual(data["secure_h"], 1.44) # Damped from 1.9

    def test_qhttp_transmit(self):
        client = TestClient(qhttp_app)
        headers = {
            "x-fractal-dimension": "1.618",
            "x-entanglement-id": "ent-123"
        }
        response = client.post("/transmit", json={"msg": "hello"}, headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["delivery"], "INSTANTANEOUS")
        self.assertEqual(data["header_echo"]["dimension"], 1.618)
        self.assertEqual(data["header_echo"]["id"], "ent-123")

    def test_qhttp_transmit_unsafe(self):
        client = TestClient(qhttp_app)
        headers = {"x-fractal-dimension": "2.5"}
        response = client.post("/transmit", json={"msg": "hello"}, headers=headers)
        data = response.json()
        self.assertEqual(data["header_echo"]["dimension"], 1.44) # Safety check applied

    def test_starlink_emanate(self):
        client = TestClient(starlink_app)
        response = client.post("/emanate")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["action"], "GLOBAL_DREAM_SYNC")
        self.assertEqual(data["parameters"]["damping"], 0.6)

    def test_safety_guard(self):
        self.assertEqual(safety_check(1.5), 1.5)
        self.assertEqual(safety_check(2.0), 1.44)
        self.assertEqual(safety_check(1.0), 1.618)

if __name__ == '__main__':
    unittest.main()
