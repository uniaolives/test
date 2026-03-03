#!/usr/bin/env python3
# tests/asi/test_physical_network.py
import sys
import os
import unittest
import numpy as np

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from asi.physics.relativistic_sim import RelativisticCompensator
from asi.network.honeycomb_435 import Honeycomb435
from asi.network.spectral_estimator import DistributedSpectralEstimator

class TestPhysicalNetwork(unittest.TestCase):
    def test_relativistic_correction_accuracy(self):
        """Verify that relativistic residual error is within spec (< 0.1 Hz)."""
        compensator = RelativisticCompensator()
        # Mock a small constant shift scenario
        t = 0.0
        dt = 0.001
        for _ in range(1000):
            # True values
            true_doppler = 10.0 # Hz
            true_grav = 2.0 # Hz
            alt_diff = (true_grav * (compensator.c**2)) / (compensator.laser_freq * compensator.g)

            # Estimate
            current_est = compensator.doppler_correction + compensator.grav_correction
            phase_err = (true_doppler + true_grav) - current_est

            compensator.update(t, phase_err, alt_diff)
            t += dt

        final_err = abs((true_doppler + true_grav) - (compensator.doppler_correction + compensator.grav_correction))
        print(f"Final Relativistic Error: {final_err:.6f} Hz")
        self.assertTrue(final_err < 0.1, f"Error {final_err} exceeds 0.1 Hz spec")

    def test_honeycomb_adjacency(self):
        """Verify honeycomb degree distribution."""
        h = Honeycomb435(layers=1)
        # In a 1-layer honeycomb centered at node 0, node 0 has degree 4.
        deg_0 = h.graph.degree(0)
        self.assertEqual(deg_0, 4)
        # All neighbors in layer 1 have degree 1 (leaves in our generation)
        for n in h.graph.neighbors(0):
            self.assertEqual(h.graph.degree(n), 1)

    def test_spectral_convergence(self):
        """Verify that λ2 estimation converges for a small graph."""
        num_nodes = 5
        adj = np.zeros((num_nodes, num_nodes))
        # Complete graph K5
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                adj[i,j] = adj[j,i] = 1

        # Theoretical λ2 for K_n is n
        theoretical = 5.0

        estimator = DistributedSpectralEstimator(num_nodes, adj)
        for _ in range(100):
            estimator.iterate()

        l2 = estimator.estimate_lambda2()
        print(f"Estimated λ2 (K5): {l2:.4f} (Target: {theoretical})")
        self.assertAlmostEqual(l2, theoretical, places=1)

if __name__ == "__main__":
    unittest.main()
