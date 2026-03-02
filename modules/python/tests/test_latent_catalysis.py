import unittest
import numpy as np
from modules.python.latent_catalysis_sim import (
    BITS_85,
    check_conway_norton,
    calculate_moduli_mass,
    LatentKineticsSimulator
)

class TestLatentCatalysis(unittest.TestCase):
    def test_moonshine_key(self):
        # Verifica se a chave de 85 bits satisfaz a primeira identidade (Delta_n4)
        results = check_conway_norton(BITS_85)
        self.assertIn('Delta_n4', results)
        self.assertEqual(results['Delta_n4'], 0) # Conforme análise anterior

    def test_moduli_mass(self):
        # Verifica se a massa calculada para (24, 24) está na escala GUT/Planck
        m_eff = calculate_moduli_mass(24, 24)
        self.assertGreater(m_eff, 1e10)
        self.assertLess(m_eff, 1e12)

    def test_catalysis_gain(self):
        t = np.linspace(0, 1000, 10)
        sim = LatentKineticsSimulator(t)
        results_std = sim.simulate(use_latent_catalysis=False)
        results_lat = sim.simulate(use_latent_catalysis=True)

        gain = results_lat['Acetamide'][-1] / (results_std['Acetamide'][-1] + 1e-20)
        self.assertGreater(gain, 1.0)
        print(f"Test Catalysis Gain: {gain:.2f}x")

if __name__ == "__main__":
    unittest.main()
