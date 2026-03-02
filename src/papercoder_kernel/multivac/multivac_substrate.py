from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np

@dataclass
class ComputeNode:
    """Represents a physical or biological node in the Multivac substrate."""
    node_id: str
    compute_capacity: float  # e.g., in PFLOPS
    memory: float           # e.g., in TB or PB
    coherence: float        # [0, 1]
    location: Tuple[float, float, float]
    node_type: str          # 'cloud', 'edge', 'quantum', 'biological'
    local_time: float = 0.0  # Time coordinate for the node

@dataclass
class HyperEdge:
    """Represents a connection between nodes in the hypergraph."""
    edge_id: str
    source: str              # node_id
    target: str              # node_id
    weight: float = 1.0
    type: str = 'standard'   # 'standard', 'temporal', 'holographic'
    handover_duration: float = 0.0

class MultivacSubstrate:
    """
    Substrate management for Multivac nodes.
    Tracks global capacity, coherence, and entropy.
    """
    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.edges: List[HyperEdge] = []

    def register_node(self, node: ComputeNode):
        """Registers a node into the substrate."""
        self.nodes[node.node_id] = node

    @property
    def total_capacity(self) -> float:
        """Total compute capacity across all nodes."""
        return sum(node.compute_capacity for node in self.nodes.values())

    @property
    def global_coherence(self) -> float:
        """Calculates global system coherence."""
        if not self.nodes:
            return 0.0
        # Simple mean coherence for simulation purposes
        return float(np.mean([node.coherence for node in self.nodes.values()]))

    def measure_entropy(self) -> float:
        """
        Measures system entropy based on coherence distribution.
        Lower coherence variance and higher mean suggest lower entropy in this model.
        """
        coherences = np.array([node.coherence for node in self.nodes.values()])
        if len(coherences) < 2:
            return 1.0

        # Simple heuristic for simulation entropy
        # Based on how much nodes deviate from maximum coherence
        return float(np.mean(1.0 - coherences))

    def allocate_computation(self, complexity: float, required_coherence: float) -> List[str]:
        """
        Allocates a subset of nodes meeting the coherence requirement.
        Selection is proportional to the query complexity.
        """
        eligible_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.coherence >= required_coherence
        ]

        if not eligible_nodes:
            return []

        # Limit nodes based on complexity to simulate resource management
        num_to_allocate = max(1, min(len(eligible_nodes), int(complexity * 100)))
        return eligible_nodes[:num_to_allocate]

    def add_edge(self, edge: HyperEdge):
        """Adds an edge to the hypergraph."""
        self.edges.append(edge)

    def bfs_from_node(self, start_node_id: str, max_depth: int = 10) -> List[str]:
        """Returns a list of reachable node IDs from the start node."""
        if start_node_id not in self.nodes:
            return []

        visited = {start_node_id}
        queue = [(start_node_id, 0)]
        reachable = []

        # Build adjacency list for efficiency
        adj = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            adj[edge.source].append(edge.target)

        while queue:
            current_id, depth = queue.pop(0)
            reachable.append(current_id)

            if depth < max_depth:
                for neighbor in adj.get(current_id, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        return reachable

    def find_path(self, start_id: str, target_id: str) -> Optional[List[str]]:
        """Finds a path between two nodes using BFS."""
        if start_id not in self.nodes or target_id not in self.nodes:
            return None

        if start_id == target_id:
            return [start_id]

        visited = {start_id: None}
        queue = [start_id]

        adj = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            adj[edge.source].append(edge.target)

        while queue:
            current = queue.pop(0)
            if current == target_id:
                path = []
                while current:
                    path.append(current)
                    current = visited[current]
                return path[::-1]

            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)

        return None
