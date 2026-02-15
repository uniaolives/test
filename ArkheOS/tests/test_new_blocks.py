import unittest
import numpy as np
from arkhe.ucd import UCD
from arkhe.projections import effective_dimension
from arkhe.arkhen_11_unified import Arkhen11
from arkhe.arkhe_rfid import VirtualDeviceNode
from arkhe.divergence import DivergenceProtocol
from arkhe.fusion import FusionEngine, FibonacciGeodesic
from arkhe.atmospheric import SpriteEvent, VanAllenMemory

class TestArkheFramework(unittest.TestCase):
    def test_ucd_conservation(self):
        data = np.random.rand(10, 5)
        ucd = UCD(data)
        res = ucd.analyze()
        self.assertTrue(res['conservation'])

    def test_arkhen_11_unified(self):
        arkhen = Arkhen11()
        self.assertEqual(len(arkhen.nodes), 11)
        d_eff = arkhen.effective_dimension(lambda_reg=1.0)
        self.assertGreater(d_eff, 0)
        self.assertLessEqual(d_eff, 11)

    def test_virtual_device_anomalies(self):
        device = VirtualDeviceNode("G1", "Device", (0, 0))
        device.simulate_anomaly("Leitura Perdida")
        self.assertEqual(device.coherence_history[-1]['C'], 0.35)

    def test_divergence_protocols(self):
        device = VirtualDeviceNode("G2", "Device", (0, 0))
        dp = DivergenceProtocol(device)
        dp.execute_protocol("SACRIFICE")
        self.assertEqual(device.status, "Inerte")

    def test_fusion_engine(self):
        engine = FusionEngine(lambda_reg=0.1)
        res = engine.execute_fusion(fuel_c=0.9)
        self.assertGreater(res['energy'], 0)
        self.assertAlmostEqual(res['coherence'] + res['fluctuation'], 1.0)

    def test_fibonacci_geodesic(self):
        geo = FibonacciGeodesic()
        path = geo.generate_path(steps=10)
        self.assertEqual(path.shape, (10, 2))
        # Check if radius increases (it's an outward spiral)
        self.assertGreater(path[-1, 0], path[0, 0])

    def test_atmospheric_events(self):
        sprite = SpriteEvent()
        memory = VanAllenMemory()
        light = sprite.trigger(0.8)
        self.assertEqual(light, 0.8)
        bits = memory.capture_excess("Test", 10.0)
        self.assertEqual(bits, 1.0)
        self.assertEqual(memory.total_satoshi, 1.0)

if __name__ == "__main__":
    unittest.main()
