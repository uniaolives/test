import unittest
import numpy as np
from arkhe.ucd import UCD
from arkhe.projections import effective_dimension
from arkhe.arkhen_11_unified import Arkhen11
from arkhe.arkhe_rfid import VirtualDeviceNode
from arkhe.divergence import DivergenceProtocol
from arkhe.fusion import FusionEngine, FibonacciGeodesic
from arkhe.atmospheric import SpriteEvent, VanAllenMemory
from arkhe.semidirac import SemiDiracFermion
from arkhe.lyrics import LyricalAnalyzer, get_harmony_chaos_poem

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

    def test_semidirac_properties(self):
        fermion = SemiDiracFermion()
        # E(px, 0) should be quadratic: px²/2m
        E_x = fermion.get_dispersion(0.5, 0.0)
        self.assertAlmostEqual(E_x, (0.5**2 / 2.0))
        # E(0, py) should be linear: v|py|
        E_y = fermion.get_dispersion(0.0, 0.5)
        self.assertAlmostEqual(E_y, 0.5)
        self.assertTrue(fermion.verify_tensor_conservation())

    def test_lyrical_analysis(self):
        poem = get_harmony_chaos_poem()
        analyzer = LyricalAnalyzer(poem)
        analysis = analyzer.analyze_structure()
        self.assertIn("Mirrored lines", analysis)
        self.assertIn("Chaos is dominant", analysis)

if __name__ == "__main__":
    unittest.main()
