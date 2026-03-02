import unittest
import numpy as np
from merkabah.quantum.wang_2026 import SidebandTeleportationAsArkheHandover, HybridArkheNode

class TestWang2026(unittest.TestCase):
    def setUp(self):
        self.demo = SidebandTeleportationAsArkheHandover(base_frequency=5e6)
        self.node = HybridArkheNode()

    def test_sideband_selection_case_i(self):
        # Case I: φ = π for 5 MHz -> τ = 100 ns
        tau = 1 / (2 * 5e6)
        modes = self.demo.select_teleportable_modes('I', tau)

        # Expected: 5, 15, 25 MHz (odd multiples)
        self.assertIn(5.0, modes)
        self.assertIn(15.0, modes)
        self.assertIn(25.0, modes)
        self.assertNotIn(10.0, modes)
        self.assertNotIn(20.0, modes)

    def test_sideband_selection_case_ii(self):
        # Case II: φ = 0 for 5 MHz -> τ = 0
        tau = 0
        modes = self.demo.select_teleportable_modes('II', tau)

        # Expected: 5, 10, 15, 20, 25 MHz (all meet cos(0)=1 condition)
        # Wait, if tau=0, then phi=0 for ALL frequencies.
        # cos(0) = 1, so all are selected for Case II.
        self.assertIn(5.0, modes)
        self.assertIn(10.0, modes)
        self.assertIn(15.0, modes)
        self.assertIn(20.0, modes)
        self.assertIn(25.0, modes)

    def test_demonstration_output(self):
        results = self.demo.demonstrate_wang_2026()
        self.assertEqual(results['fidelity'], 0.70)
        self.assertEqual(results['coherence_regime'], 'quantum')
        self.assertTrue(len(results['case_I']) >= 1)

    def test_hybrid_node_routing(self):
        # Quantum handover
        req_q = {'requires_quantum_fidelity': True, 'qumodes': [1, 2, 3]}
        res_q = self.node.process_handover(req_q)
        self.assertEqual(res_q, [1, 2, 3])

        # Classical handover
        req_c = {'requires_quantum_fidelity': False, 'data': 'Some data'}
        res_c = self.node.process_handover(req_c)
        self.assertEqual(res_c, "Transmitted classically: Some data")

if __name__ == '__main__':
    unittest.main()
