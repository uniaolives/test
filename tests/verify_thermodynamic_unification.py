# tests/verify_thermodynamic_unification.py
import sys
import os
import unittest
import numpy as np

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.python.arkhe_physics.entropy_unit import ArkheEntropyUnit
from modules.python.secops.phi_anomaly_detector import PhiAnomalyDetector

class TestThermodynamicUnification(unittest.TestCase):
    def test_aeu_conversions(self):
        # 1 bit ≈ 0.693 k_B T J/K
        aeu = ArkheEntropyUnit(value=1.0, domain='informational', context='test')
        bits = aeu.to_informational()
        self.assertEqual(bits, 1.0)

        physical = aeu.to_physical(temperature=300.0)
        k_B = 1.380649e-23
        self.assertAlmostEqual(physical, k_B * np.log(2), places=30)

    def test_phi_anomaly_detection(self):
        detector = PhiAnomalyDetector()

        # Handover normal em Engenharia
        handover_ok = {
            'id': 'h1',
            'source_layer': 'engineering',
            'energy_j': 1e-4,
            'duration_ms': 100,
            'timestamp': 1000
        }
        # phi = 1e-4 / 100 = 1e-6 (dentro do limite [1e-6, 1e-2])
        self.assertIsNone(detector.analyze_handover(handover_ok))

        # Handover anômalo (eficiência impossível)
        handover_anomaly = {
            'id': 'h2',
            'source_layer': 'engineering',
            'energy_j': 1.0,
            'duration_ms': 1,
            'timestamp': 1001
        }
        # phi = 1.0 / 1 = 1.0 (fora do limite high=1e-2)
        alert = detector.analyze_handover(handover_anomaly)
        self.assertIsNotNone(alert)
        self.assertEqual(alert['reason'], 'EFICIÊNCIA IMPOSSÍVEL')

if __name__ == "__main__":
    unittest.main()
