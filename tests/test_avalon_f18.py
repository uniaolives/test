
import unittest
import numpy as np
from avalon.analysis.fractal import FractalAnalyzer, calculate_adaptive_hausdorff, FractalSecurityError

class TestFractalSecurity(unittest.TestCase):
    def test_adaptive_hausdorff_damping(self):
        # Verify that h decreases with iterations (damping)
        h0 = calculate_adaptive_hausdorff(0)
        h100 = calculate_adaptive_hausdorff(100)
        self.assertLess(h100, h0)
        self.assertGreater(h100, 1.0)

    def test_iteration_limit(self):
        with self.assertRaises(FractalSecurityError):
            calculate_adaptive_hausdorff(1001)

    def test_analyzer_coherence_damping(self):
        analyzer = FractalAnalyzer(damping=0.6)
        # Create a very noisy signal to trigger low coherence
        signal = np.random.normal(0, 10, 1000)
        result = analyzer.analyze(signal)

        # Damping should have increased if coherence was low
        self.assertGreaterEqual(analyzer.damping, 0.6)
        self.assertEqual(result['f18_status'], 'COMPLIANT')

if __name__ == '__main__':
    unittest.main()
