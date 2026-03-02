import unittest
from asi.core.hypergraph import Hypergraph
from asi.domains import neuroscience, nanotechnology

class TestASICore(unittest.TestCase):
    def test_hypergraph_basic(self):
        h = Hypergraph()
        n1 = h.add_node("n1")
        n2 = h.add_node("n2")
        h.add_edge({"n1", "n2"}, weight=0.8)
        h.bootstrap_step()
        self.assertEqual(h.total_coherence(), 0.8)

    def test_neuroscience_simulation(self):
        h = Hypergraph()
        neuroscience.simulate_place_cells(h, num_cells=10, positions=10)
        # With positions=10 and num_cells=10, overlap should be high
        self.assertGreater(h.total_coherence(), 0.0)

    def test_nanotechnology_simulation(self):
        h = Hypergraph()
        nanotechnology.simulate_ucnp(h, trigger=True)
        self.assertGreater(h.total_coherence(), 0.0)

if __name__ == "__main__":
    unittest.main()
