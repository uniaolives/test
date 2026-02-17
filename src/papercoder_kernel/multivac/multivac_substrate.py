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

class MultivacSubstrate:
    """
    Substrate management for Multivac nodes.
    Tracks global capacity, coherence, and entropy.
    """
    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}

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
