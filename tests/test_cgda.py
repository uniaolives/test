import unittest
import asyncio
import numpy as np
from cosmos.cgda import CGDALab, ConstraintMethod

class TestCGDA(unittest.IsolatedAsyncioTestCase):

    async def test_cgda_lab_ingestion(self):
        lab = CGDALab("test_lab")
        states = [{'id': 's1', 'features': [1, 0], 'probability': 1.0}]
        lab.load_observed_states(states)
        self.assertIn('s1', lab.observed_states)

    async def test_cgda_derivation_full(self):
        lab = CGDALab("test_lab")
        states = [
            {'id': 's1', 'features': [1, 0, 0], 'probability': 0.5},
            {'id': 's2', 'features': [0, 1, 0], 'probability': 0.5}
        ]
        lab.load_observed_states(states)
        geometry = await lab.derive_constraint_geometry(ConstraintMethod.FULL)
        self.assertIsNotNone(geometry.constraint_matrix)
        # Should have detected the 3rd dimension as constrained
        self.assertEqual(geometry.allowed_subspace.shape[1], 2)

    async def test_ising_ingestion(self):
        lab = CGDALab("test_lab")
        ising_data = [{'id': 'm1', 'ground_states': [[1, 1], [-1, -1]]}]
        lab.ingest_ising_model(ising_data)
        self.assertIn('ising_m1_gs_0', lab.observed_states)

if __name__ == "__main__":
    unittest.main()
