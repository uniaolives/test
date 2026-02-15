import unittest
import numpy as np
from arkhe.ucd import UCD, verify_conservation
from arkhe.projections import effective_dimension
from arkhe.arkhen_11 import Arkhen11
from arkhe.arkhe_rfid import RFIDTag

class TestNewBlocks(unittest.TestCase):
    def test_ucd_conservation(self):
        data = np.random.rand(10, 5)
        ucd = UCD(data)
        res = ucd.analyze()
        self.assertTrue(res['conservation'])
        self.assertAlmostEqual(res['C'] + res['F'], 1.0)

    def test_effective_dimension(self):
        F = np.eye(5)
        d_eff, _ = effective_dimension(F, 1.0)
        # tr(I * (I + I)^-1) = tr(0.5 * I) = 2.5
        self.assertAlmostEqual(d_eff, 2.5)

    def test_arkhen_11_coherence(self):
        arkhen = Arkhen11()
        C = arkhen.compute_coherence()
        self.assertGreater(C, 0)
        self.assertLessEqual(C, 1.0)

    def test_rfid_conservation(self):
        tag = RFIDTag("T1", "Object")
        tag.read("R1", "L1")
        import time
        time.sleep(0.1)
        tag.read("R2", "L2")
        self.assertTrue(tag.verify_conservation())

if __name__ == "__main__":
    unittest.main()
