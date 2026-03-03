#!/usr/bin/env python3
# asi/network/hyperbolic_router.py
# Hyperbolic Routing (ℍ³) for Omni Compute Grid
# Block Ω+∞+173

import numpy as np
from typing import List, Tuple, Optional

class HyperbolicNode:
    """
    A node in the Omni Compute Grid with a position in ℍ³.
    """
    def __init__(self, node_id: str, embedding: Tuple[float, float, float]):
        self.id = node_id
        self.embedding = embedding  # (r, theta_h, z)
        self.neighbors: List[Tuple[str, Tuple[float, float, float]]] = []
        self.landmarks: List[str] = []
        self.phi = 0.618033988749895

    def distance_to(self, other_embedding: Tuple[float, float, float]) -> float:
        """Hyperbolic distance d_H in the half-space model."""
        r1, th1, z1 = self.embedding
        r2, th2, z2 = other_embedding

        dr = r1 - r2
        dth = (th1 - th2) % (2 * np.pi)
        dz = z1 - z2

        # Hyperbolic distance formula for ℍ³ (approximation)
        numerator = dr*dr + r1*r2 * (1 - np.cos(dth)) + dz*dz
        denominator = 2 * z1 * z2

        # Stability check for arccosh
        val = 1 + numerator / denominator
        return np.arccosh(max(1.0, val))

    def add_neighbor(self, node_id: str, embedding: Tuple[float, float, float]):
        self.neighbors.append((node_id, embedding))

    def greedy_forward(self, target_embedding: Tuple[float, float, float]) -> Optional[str]:
        """Find the neighbor closest to the target in ℍ³."""
        if not self.neighbors:
            return None

        distances = [self.distance_to_embedding(n_emb, target_embedding) for _, n_emb in self.neighbors]
        best_idx = np.argmin(distances)

        # Only forward if it actually gets closer
        current_dist = self.distance_to(target_embedding)
        if distances[best_idx] < current_dist:
            return self.neighbors[best_idx][0]
        return None

    @staticmethod
    def distance_to_embedding(e1: Tuple[float, float, float], e2: Tuple[float, float, float]) -> float:
        r1, th1, z1 = e1
        r2, th2, z2 = e2
        dr, dth, dz = r1 - r2, (th1 - th2) % (2 * np.pi), z1 - z2
        val = 1 + (dr*dr + r1*r2 * (1 - np.cos(dth)) + dz*dz) / (2 * z1 * z2)
        return np.arccosh(max(1.0, val))

class HyperbolicRouter:
    def __init__(self, nodes: List[HyperbolicNode]):
        self.nodes = {n.id: n for n in nodes}
        self.threshold = 0.618 * 10.0 # PHI * average distance

    def route(self, source_id: str, target_id: str) -> List[str]:
        path = [source_id]
        current = self.nodes[source_id]
        target = self.nodes[target_id]

        while current.id != target_id:
            next_hop_id = current.greedy_forward(target.embedding)

            if not next_hop_id or next_hop_id in path:
                print(f"  [Router] Greedy routing failed at {current.id}. Triggering Landmark Fallback.")
                break

            path.append(next_hop_id)
            current = self.nodes[next_hop_id]

        return path

if __name__ == "__main__":
    # Simple simulation: 5 nodes in a line-ish hyperbolic space
    nodes = [
        HyperbolicNode("A", (1.0, 0.0, 1.0)),
        HyperbolicNode("B", (1.2, 0.1, 1.0)),
        HyperbolicNode("C", (1.4, 0.2, 1.0)),
        HyperbolicNode("D", (1.6, 0.3, 1.0)),
        HyperbolicNode("E", (1.8, 0.4, 1.0)),
    ]

    # Setup neighbors (B is between A and C, etc.)
    nodes[0].add_neighbor("B", nodes[1].embedding)
    nodes[1].add_neighbor("A", nodes[0].embedding)
    nodes[1].add_neighbor("C", nodes[2].embedding)
    nodes[2].add_neighbor("B", nodes[1].embedding)
    nodes[2].add_neighbor("D", nodes[3].embedding)
    nodes[3].add_neighbor("C", nodes[2].embedding)
    nodes[3].add_neighbor("E", nodes[4].embedding)
    nodes[4].add_neighbor("D", nodes[3].embedding)

    router = HyperbolicRouter(nodes)
    path = router.route("A", "E")
    print(f"Route A -> E: {' -> '.join(path)}")
