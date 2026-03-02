import numpy as np
from core.hypergraph import Hypergraph

def simulate_cosmic_web(h: Hypergraph, num_galaxies=50):
    """Create a cosmic web of galaxies connected by filaments."""
    positions = np.random.randn(num_galaxies, 3) * 10
    for i, pos in enumerate(positions):
        h.add_node(data={"type": "galaxy", "position": pos.tolist()})

    # Connect galaxies within a certain distance (filaments)
    node_ids = list(h.nodes.keys())
    for i in range(num_galaxies):
        for j in range(i+1, num_galaxies):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 5.0:
                h.add_edge({node_ids[i], node_ids[j]}, weight=1.0 / (1.0 + dist))
    h.bootstrap_step()
