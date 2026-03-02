import numpy as np
from core.hypergraph import Hypergraph

def simulate_place_cells(h: Hypergraph, num_cells: int = 10, positions: int = 100) -> None:
    """Simulate place cells tuned to positions along a 1D track."""
    for i in range(num_cells):
        node = h.add_node(data={"type": "place_cell", "preferred_position": i * (positions/num_cells)})
    # Create edges between cells with overlapping fields (like‑to‑like)
    for i, n1 in enumerate(list(h.nodes.values())[:num_cells]):
        for j, n2 in enumerate(list(h.nodes.values())[i+1:num_cells]):
            pos1 = n1.data["preferred_position"]
            pos2 = n2.data["preferred_position"]
            overlap = max(0, 1 - abs(pos1-pos2)/(positions/num_cells))
            if overlap > 0.5:
                h.add_edge({n1.id, n2.id}, weight=overlap)
    h.bootstrap_step()

def simulate_like_to_like_attention(h: Hypergraph, num_neurons: int = 20, feature_dims: int = 2) -> None:
    """Simulate attention mechanism where neurons with similar features connect."""
    # Generate random feature vectors (orientation tuning)
    features = np.random.randn(num_neurons, feature_dims)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    for i in range(num_neurons):
        h.add_node(data={"type": "neuron", "feature_vector": features[i].tolist()})

    # Connect based on cosine similarity
    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            sim = np.dot(features[i], features[j])
            if sim > 0.7:
                h.add_edge({list(h.nodes.keys())[i], list(h.nodes.keys())[j]}, weight=sim)
    h.bootstrap_step()
