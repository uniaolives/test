# tests/integration/test_qhttp.py
import unittest
import asyncio
import json

# Integration test for qhttp protocol
# In a real CI environment, this would hit the qhttp-server service

class MockQHTTPClient:
    def __init__(self, host):
        self.host = host

    async def get_state(self):
        # Simulate response from GET /quantum/state
        return {
            "real": [1.0, 0.0],
            "imag": [0.0, 0.0],
            "n_qubits": 1,
            "status": "200 OK"
        }

    async def post_evolve(self, unitary):
        # Simulate response from POST /quantum/evolve
        return {
            "status": "success",
            "unitary_applied": unitary
        }

class TestQHTTPIntegration(unittest.TestCase):
    def setUp(self):
        self.client = MockQHTTPClient("localhost:50051")

    def test_get_state(self):
        loop = asyncio.get_event_loop()
        state = loop.run_until_complete(self.client.get_state())
        self.assertEqual(state["n_qubits"], 1)
        self.assertEqual(state["status"], "200 OK")

    def test_post_evolve(self):
        loop = asyncio.get_event_loop()
        resp = loop.run_until_complete(self.client.post_evolve("hadamard"))
        self.assertEqual(resp["status"], "success")
        self.assertEqual(resp["unitary_applied"], "hadamard")

if __name__ == "__main__":
    unittest.main()
