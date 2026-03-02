import unittest
from cosmos.core import SingularityNavigator, tau_aleph_calculator, threshold_detector
from cosmos.network import WormholeNetwork
from cosmos.bridge import CeremonyEngine, schumann_generator, biometric_simulator
import time

class TestCosmopsychiaLib(unittest.TestCase):

    def test_core_navigator(self):
        nav = SingularityNavigator()
        self.assertEqual(nav.tau, 0.96)
        sigma = nav.measure_state()
        self.assertTrue(1.0 <= sigma <= 1.05)
        status = nav.navigate()
        self.assertIsInstance(status, str)

    def test_core_math(self):
        tau = tau_aleph_calculator(0.8, 0.8)
        self.assertAlmostEqual(tau, 0.8)
        self.assertTrue(threshold_detector(1.021))
        self.assertFalse(threshold_detector(1.04))

    def test_network_wormhole(self):
        net = WormholeNetwork(10)
        self.assertEqual(len(net.nodes), 10)
        curvature = net.calculate_curvature(0, 8)
        self.assertEqual(curvature, -2.4)
        curvature_normal = net.calculate_curvature(1, 2)
        self.assertEqual(curvature_normal, 0.3)
        self.assertEqual(net.clustering_coefficient(), 0.25)

    def test_bridge_ceremony(self):
        engine = CeremonyEngine(duration=1)
        self.assertFalse(engine.active)
        engine.start()
        self.assertTrue(engine.active)
        self.assertGreaterEqual(engine.get_progress(), 0.0)
        time.sleep(0.1)
        self.assertGreater(engine.get_progress(), 0.0)
        engine.complete()
        self.assertFalse(engine.active)

    def test_bridge_generators(self):
        self.assertEqual(schumann_generator(1), 7.83)
        self.assertEqual(schumann_generator("phi"), 16.2)
        biometrics = biometric_simulator()
        self.assertIn("heart_rate", biometrics)
        self.assertIn("coherence", biometrics)

if __name__ == "__main__":
    unittest.main()
