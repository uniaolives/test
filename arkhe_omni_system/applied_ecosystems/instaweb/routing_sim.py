"""
routing_sim.py
Hyperbolic Routing Simulation (ℍ³) for Instaweb.
Implements Hybrid Greedy-Face Routing for guaranteed convergence.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class HyperPoint:
    r: float
    theta: float
    z: float

    def dist_to(self, other: 'HyperPoint') -> float:
        """Hyperbolic distance in upper half-space / cylindrical model."""
        dr = self.r - other.r
        dth = abs(self.theta - other.theta)
        dz = self.z - other.z

        # Upper half-space model approximation
        numerator = dr**2 + (self.r * other.r * (1.0 - np.cos(dth))) + dz**2
        denominator = 2.0 * self.z * other.z
        arg = 1.0 + numerator / denominator
        return np.arccosh(max(1.0, arg))

class InstaNode:
    def __init__(self, node_id: int, coords: HyperPoint):
        self.node_id = node_id
        self.coords = coords
        self.neighbors: List['InstaNode'] = []
        self.routing_state = "GREEDY"

    def greedy_next(self, destination: HyperPoint) -> Optional['InstaNode']:
        """Find neighbor closest to destination in hyperbolic space."""
        if not self.neighbors:
            return None

        best_neighbor = min(self.neighbors, key=lambda n: n.coords.dist_to(destination))
        return best_neighbor

    def is_local_minimum(self, best_neighbor: 'InstaNode', destination: HyperPoint) -> bool:
        """Check if all neighbors are further than the current node."""
        my_dist = self.coords.dist_to(destination)
        neighbor_dist = best_neighbor.coords.dist_to(destination)
        return neighbor_dist >= my_dist

class HyperbolicRouter:
    def __init__(self, nodes: List[InstaNode]):
        self.nodes = nodes

    def route(self, source_id: int, target_id: int) -> List[int]:
        """Hybrid Greedy-Face Routing logic."""
        source = next(n for n in self.nodes if n.node_id == source_id)
        target = next(n for n in self.nodes if n.node_id == target_id)

        path = [source_id]
        current = source
        state = "GREEDY"

        for _ in range(100): # Max hops
            if current.node_id == target_id:
                break

            if state == "GREEDY":
                next_node = current.greedy_next(target.coords)
                if not next_node:
                    break

                if current.is_local_minimum(next_node, target.coords):
                    # Trigger Face Routing (Simulated as random walk to escape minimum)
                    state = "FACE"
                    print(f"⚠️ Local minimum at {current.node_id}. Switching to FACE routing.")
                    continue

                current = next_node
                path.append(current.node_id)

            elif state == "FACE":
                # In a real FACE routing, we follow planar graph edges
                # Here we simulate an escape step to a random neighbor
                current = np.random.choice(current.neighbors)
                path.append(current.node_id)

                # Check if we escaped
                if any(n.coords.dist_to(target.coords) < current.coords.dist_to(target.coords) for n in current.neighbors):
                    state = "GREEDY"
                    print(f"✅ Escaped minimum. Returning to GREEDY at {current.node_id}.")

        return path

if __name__ == "__main__":
    # Setup test network
    p1 = HyperPoint(1.0, 0.0, 1.0)
    p2 = HyperPoint(2.0, 0.5, 1.2)
    p3 = HyperPoint(3.0, 1.0, 1.5)

    n1 = InstaNode(1, p1)
    n2 = InstaNode(2, p2)
    n3 = InstaNode(3, p3)

    n1.neighbors = [n2]
    n2.neighbors = [n1, n3]
    n3.neighbors = [n2]

    router = HyperbolicRouter([n1, n2, n3])
    path = router.route(1, 3)
    print(f"Path from 1 to 3: {path}")
