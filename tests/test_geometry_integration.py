# tests/test_geometry_integration.py
import unittest
from core.python.axos.axos_v3 import AxosV3
import numpy as np

class TestGeometryIntegration(unittest.TestCase):
    def setUp(self):
        self.axos = AxosV3()

    def test_axos_v3_has_geometry_methods(self):
        self.assertTrue(hasattr(self.axos, 'explore_landscape'))
        self.assertTrue(hasattr(self.axos, 'generate_entity'))
        self.assertTrue(hasattr(self.axos, 'correlate_hodge'))

    def test_correlate_hodge(self):
        result = self.axos.correlate_hodge(n_samples=50)
        self.assertEqual(result.status, "SUCCESS")
        self.assertIn("regression_score", result.data)
        self.assertIn("feature_importances", result.data)

    def test_generate_entity(self):
        result = self.axos.generate_entity()
        self.assertEqual(result.status, "SUCCESS")
        self.assertIn("h11", result.data)
        self.assertIn("h21", result.data)
        self.assertIn("coherence", result.data)

    def test_explore_landscape(self):
        # This will test the RL agent integration
        result = self.axos.explore_landscape(h11=491 # safety, h21=50)
        self.assertEqual(result.status, "SUCCESS")
        self.assertIn("coherence", result.data)
        self.assertIn("new_state", result.data)

if __name__ == "__main__":
    unittest.main()
