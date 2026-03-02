import unittest
import numpy as np
from arkhe.bioelectricity import BioelectricGrid, IonChannelNode, ConductionMode

class TestBioelectricity(unittest.TestCase):
    def test_ephaptic_coupling(self):
        # Two nodes very close (within Mori limit)
        node_a = IonChannelNode('A', 'NaV', -60.0, 10, (0.0, 0.0))
        node_b = IonChannelNode('B', 'NaV', -70.0, 10, (10.0, 0.0))

        influence = ConductionMode.ephaptic(node_a, node_b)
        # Field within 30nm should be plateaued based on density
        self.assertGreater(influence, 0)

    def test_synchronization(self):
        grid = BioelectricGrid()
        # Add 3 nodes close to each other
        grid.add_node(IonChannelNode('1', 'NaV', -70.0, 10, (0.0, 0.0)))
        grid.add_node(IonChannelNode('2', 'NaV', -70.0, 10, (10.0, 0.0)))
        grid.add_node(IonChannelNode('3', 'NaV', -70.0, 10, (0.0, 10.0)))

        # Initial coherence should be low (random phases)
        avg_phase_vector = np.mean([np.exp(1j * p) for p in grid.phases.values()])
        initial_coherence = np.abs(avg_phase_vector)

        final_coherence = grid.simulate_ephaptic_sync(steps=200)

        # After sync, coherence should generally increase if nodes are close
        self.assertTrue(grid.detect_consciousness_signature() or final_coherence > initial_coherence)

    def test_mori_limit_decay(self):
        node_a = IonChannelNode('A', 'NaV', -60.0, 10, (0.0, 0.0))
        # Distance 20nm (inside plateau)
        field_near = node_a.compute_local_field(20.0)
        # Distance 100nm (outside plateau, should be decayed)
        field_far = node_a.compute_local_field(100.0)

        self.assertEqual(field_near, 8.0) # 10 * 0.8
        self.assertLess(field_far, 8.0)

if __name__ == '__main__':
    unittest.main()
