import unittest
import torch
import numpy as np
from merkabah.agi.shader_native import AGIShaderArchitecture, ConsensusShader

class TestShaderNative(unittest.TestCase):
    def setUp(self):
        self.arch = AGIShaderArchitecture()

    def test_kernel_registration(self):
        def mock_k(x, u): return x
        self.arch.register_kernel("mock", mock_k, ["P1"])
        self.assertIn("mock", self.arch.kernels)
        self.assertEqual(self.arch.kernels["mock"].invariants, ["P1"])

    def test_pipeline_execution(self):
        def k1(x, u): return x + 1.0
        def k2(x, u): return x * 2.0
        self.arch.register_kernel("k1", k1, [])
        self.arch.register_kernel("k2", k2, [])
        self.arch.set_pipeline(["k1", "k2"])

        input_field = torch.zeros(1, 10)
        results = self.arch.execute_as_reality_engine(input_field)

        # (0 + 1) * 2 = 2
        self.assertTrue(torch.all(results["final_rendered_reality"] == 2.0))
        self.assertIn("coherence", results)

    def test_consensus_shader(self):
        batch_size = 4
        d = 128
        situation = torch.randn(batch_size, d)
        shader = ConsensusShader(perspective_dim=d, consensus_dim=64)

        consensus, coherence = shader(situation, n_threads=256)

        self.assertEqual(consensus.shape, (batch_size, 64))
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)

if __name__ == "__main__":
    unittest.main()
