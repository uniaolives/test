import unittest
import numpy as np
from arkhe.ucd import UCD
from arkhe.projections import effective_dimension
from arkhe.arkhen_11_unified import Arkhen11
from arkhe.arkhe_rfid import VirtualDeviceNode
from arkhe.divergence import DivergenceProtocol

class TestArkheFramework(unittest.TestCase):
    def test_ucd_conservation(self):
        data = np.random.rand(10, 5)
        ucd = UCD(data)
        res = ucd.analyze()
        self.assertTrue(res['conservation'])

    def test_arkhen_11_unified(self):
        arkhen = Arkhen11()
        self.assertEqual(len(arkhen.nodes), 11)
        # Check effective dimension with default adjacency (10 connected to 1)
        d_eff = arkhen.effective_dimension(lambda_reg=1.0)
        self.assertGreater(d_eff, 0)
        self.assertLessEqual(d_eff, 11)

    def test_virtual_device_anomalies(self):
        device = VirtualDeviceNode("G1", "Device", (0, 0))
        device.simulate_anomaly("Leitura Perdida")
        self.assertEqual(device.coherence_history[-1]['C'], 0.35)
        self.assertIn("Leitura Perdida", device.anomalies_encountered)

    def test_divergence_protocols(self):
        device = VirtualDeviceNode("G2", "Device", (0, 0))
        dp = DivergenceProtocol(device)
        dp.execute_protocol("SACRIFICE")
        self.assertEqual(device.status, "Inerte")
        self.assertEqual(device.coherence_history[-1]['C'], 0.0)

if __name__ == "__main__":
    unittest.main()
