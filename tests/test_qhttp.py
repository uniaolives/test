import unittest
import math
from cosmos.qhttp import (
    SatelliteChannel,
    QuantumChannel,
    QuantumState,
    QHTTPRequest,
    QHTTPResponse,
    QHTTP_STATUS_CODES
)

class TestQHTTP(unittest.TestCase):

    def test_fidelity_calculation_zenith_clear(self):
        channel = SatelliteChannel(elevation_angle=90, weather_condition='clear')
        fidelity = channel.calculate_fidelity()
        # 90 deg -> math.sin(math.radians(90)) = 1.0
        # atmosphere_path = 10 / 1.0 = 10
        # loss_factor = 0.02 * 10 = 0.2
        # fidelity = 1.0 - 0.2 = 0.8
        self.assertAlmostEqual(fidelity, 0.8)

    def test_fidelity_calculation_horizon_rain(self):
        channel = SatelliteChannel(elevation_angle=5, weather_condition='rain')
        fidelity = channel.calculate_fidelity()
        # 5 deg -> sin(5 deg) approx 0.087
        # atmosphere_path approx 10 / 0.087 approx 114.7
        # loss_factor = 0.02 * 114.7 + 0.9 = 2.29 + 0.9 = 3.19
        # fidelity = max(0, 1.0 - 3.19) = 0
        self.assertEqual(fidelity, 0)

    def test_entanglement_generation(self):
        channel = SatelliteChannel(elevation_angle=90, weather_condition='clear')
        pair_id, status = channel.generate_entanglement_from_orbit()
        self.assertIsNotNone(pair_id)
        self.assertEqual(status, "201 Entangled via Starlink")
        self.assertIn(pair_id, channel.entanglement_registry)
        self.assertAlmostEqual(channel.entanglement_registry[pair_id]['client'].fidelity, 0.8)

    def test_entanglement_failure_clouds(self):
        channel = SatelliteChannel(elevation_angle=45, weather_condition='cloudy')
        # sin(45) = 0.707
        # path = 14.14
        # loss = 0.02 * 14.14 + 0.4 = 0.2828 + 0.4 = 0.6828
        # fidelity = 0.3172 (which is < 0.8)
        pair_id, status = channel.generate_entanglement_from_orbit()
        self.assertIsNone(pair_id)
        self.assertEqual(status, "425 Atmospheric Turbulence")

    def test_qhttp_request(self):
        req = QHTTPRequest("QGET", "quantum://test", {"X-Quantum-Key": "abc"})
        self.assertEqual(req.method, "QGET")
        self.assertEqual(req.headers["X-Quantum-Key"], "abc")

    def test_qhttp_response(self):
        res = QHTTPResponse(425)
        self.assertEqual(res.status_code, 425)
        self.assertEqual(res.status_message, "Atmospheric Turbulence")

if __name__ == "__main__":
    unittest.main()
