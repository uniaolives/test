import unittest
import numpy as np
from acds import AnisotropicCDS, AdaptiveACDS

class TestACDS(unittest.TestCase):
    def test_initialization(self):
        acds = AnisotropicCDS(geometry='tetrahedral')
        self.assertEqual(acds.geometry, 'tetrahedral')
        self.assertEqual(acds.constraint_matrix.shape, (4, 4))

    def test_coherence(self):
        acds = AnisotropicCDS(geometry='tetrahedral')
        self.assertAlmostEqual(acds.measure_coherence(), 0.430)

    def test_adaptation(self):
        acds = AdaptiveACDS(geometry='tetrahedral')
        old_matrix = acds.constraint_matrix.copy()
        acds.adapt_geometry(performance_feedback=0.5)
        self.assertFalse(np.array_equal(old_matrix, acds.constraint_matrix))

if __name__ == '__main__':
    unittest.main()
