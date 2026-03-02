import unittest
import torch
from modules.python.merkabah_cy import MerkabahCYSystem

class TestMerkabahCY(unittest.TestCase):
    def setUp(self):
        self.config = {
            'generator': {
                'latent_dim': 512,
                'num_layers': 2,
                'h11_range': (1, 1000),
                'h21_range': (1, 1000)
            },
            'mapper': {
                'node_features': 10,
                'h21_max': 50
            }
        }
        self.merkabah = MerkabahCYSystem(self.config)

    def test_pipeline_execution(self):
        z_seed = torch.randn(1, 512)
        results = self.merkabah.run_pipeline(z_seed, iterations=5)

        self.assertIn('final_entity', results)
        self.assertIn('hodge_correlations', results)
        self.assertIn('phase_history', results)

        entity = results['final_entity']
        self.assertIn('coerência_global', entity)
        self.assertGreaterEqual(entity['coerência_global'], 0)

if __name__ == "__main__":
    unittest.main()
