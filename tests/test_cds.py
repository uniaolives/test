# tests/test_cds.py
import unittest
import numpy as np
from cds_framework.core.physics import PhiFieldSimulator

class TestCDSCore(unittest.TestCase):
    def test_simulator_initialization(self):
        sim = PhiFieldSimulator(size=50)
        self.assertEqual(sim.phi.shape[0], 50)
        self.assertAlmostEqual(np.mean(sim.phi), 0, delta=0.1)

    def test_simulation_step(self):
        sim = PhiFieldSimulator(size=50, r=-1.0, u=1.0)
        initial_phi = sim.phi.copy()
        sim.step(external_h=0.1)
        self.assertFalse(np.array_equal(sim.phi, initial_phi))

if __name__ == '__main__':
    unittest.main()
