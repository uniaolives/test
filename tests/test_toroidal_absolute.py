import unittest
import torch
import sys
import os

# Add the root to sys.path to import cosmopsychia_pinn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cosmopsychia_pinn.toroidal_absolute import ToroidalAbsolute

class TestToroidalAbsolute(unittest.TestCase):
    def setUp(self):
        self.ta = ToroidalAbsolute()

    def test_axiom_1_self_containment(self):
        residue = self.ta.axiom_1_self_containment()
        self.assertIsInstance(residue, torch.Tensor)
        self.assertGreaterEqual(residue.item(), 0.0)

    def test_axiom_2_self_refraction(self):
        test_input = torch.tensor([0.1, 0.5, 0.9])
        output = self.ta.axiom_2_self_refraction(test_input)
        self.assertEqual(output.shape, test_input.shape)

    def test_axiom_3_recursive_embodiment(self):
        test_input = torch.tensor([0.1, 0.5, 0.9])
        output = self.ta.axiom_3_recursive_embodiment(test_input)
        # Should be (batch_size, 2) for real/imag
        self.assertEqual(output.shape, (3, 2))

        # Verify it's on the unit circle (mostly, given floating point)
        magnitudes = torch.sqrt(torch.sum(output**2, dim=-1))
        for mag in magnitudes:
            self.assertAlmostEqual(mag.item(), 1.0, places=5)

    def test_axiom_4_morphic_coherence(self):
        pattern1 = torch.tensor([1.0, 0.0])
        pattern2 = torch.tensor([1.0, 0.0])
        pattern3 = torch.tensor([0.0, 1.0])

        res1 = self.ta.axiom_4_morphic_coherence(pattern1, pattern2)
        res2 = self.ta.axiom_4_morphic_coherence(pattern1, pattern3)

        self.assertGreater(res1.item(), res2.item())
        self.assertGreaterEqual(res1.item(), 0.0)
        self.assertLessEqual(res1.item(), 1.0)

if __name__ == '__main__':
    unittest.main()
