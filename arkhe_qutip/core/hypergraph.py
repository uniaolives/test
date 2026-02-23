"""
ArkheHypergraph: Collection of quantum nodes and handovers.
Implements the hypergraph structure from Arkhe(n) Language.
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from qutip import Qobj
from .arkhe_qobj import ArkheQobj
from .handover import QuantumHandover

class ArkheHypergraph:
    """
    Quantum hypergraph consisting of ArkheQobj nodes and handovers.

    Parameters
    ----------
    name : str, optional
        Name of the hypergraph.
    """

    def __init__(self, name="ArkheHypergraph"):
        self.name = name
        self.nodes: Dict[str, ArkheQobj] = {}
        self.handovers: Dict[str, QuantumHandover] = {}
        self.adjacency: Dict[str, List[str]] = {}  # node_id -> list of neighbor ids

        # Global metrics
        self.global_coherence = 1.0
        self.global_phi = 0.0  # Integrated information
        self.history = []  # list of (timestamp, event, metrics)

    def add_node(self, node: ArkheQobj) -> 'ArkheHypergraph':
        """Add a node to the hypergraph."""
        self.nodes[node.node_id] = node
        self.adjacency[node.node_id] = []
        return self

    def add_handover(self, handover: QuantumHandover) -> 'ArkheHypergraph':
        """Add a handover to the hypergraph."""
        self.handovers[handover.handover_id] = handover

        # Update adjacency
        src_id = handover.source.node_id
        tgt_id = handover.target.node_id
        if tgt_id not in self.adjacency[src_id]:
            self.adjacency[src_id].append(tgt_id)
        if src_id not in self.adjacency[tgt_id]:
            self.adjacency[tgt_id].append(src_id)

        return self

    def execute_handover(self, handover_id: str) -> Optional[Qobj]:
        """
        Execute a handover by its ID.

        Parameters
        ----------
        handover_id : str
            ID of the handover to execute.

        Returns
        -------
        result : Qobj or None
            Result of the handover.
        """
        if handover_id not in self.handovers:
            raise ValueError(f"Handover {handover_id} not found")

        handover = self.handovers[handover_id]
        result = handover.execute()

        # Update global metrics
        self._update_global_metrics()

        return result

    def _update_global_metrics(self):
        """Update global coherence and integrated information."""
        if not self.nodes:
            self.global_coherence = 0.0
            self.global_phi = 0.0
            return

        # Global coherence = average of local coherences
        coherences = [node.coherence for node in self.nodes.values()]
        self.global_coherence = np.mean(coherences)

        # Integrated information Φ (simplified - average of mutual information between nodes)
        # This is a placeholder - real IIT calculation is much more complex
        total_corr = 0.0
        node_list = list(self.nodes.values())
        n = len(node_list)
        if n > 1:
            for i in range(n):
                for j in range(i+1, n):
                    # Simplified correlation based on handover frequency
                    # In practice, would need quantum mutual information
                    corr = 0.1 if j in self.adjacency[node_list[i].node_id] else 0.0
                    total_corr += corr
            self.global_phi = total_corr / (n * (n-1) / 2) if n > 1 else 0.0

        # Record history
        self.history.append({
            'timestamp': time.time(),
            'global_coherence': self.global_coherence,
            'global_phi': self.global_phi,
            'n_nodes': len(self.nodes),
            'n_handovers': len(self.handovers)
        })

    def compute_winding_number(self) -> int:
        """
        Compute total winding number (sum of individual node winding numbers).

        Returns
        -------
        int
            Total winding number.
        """
        return sum(node.winding_number for node in self.nodes.values())

    def find_path(self, start_id: str, end_id: str) -> List[str]:
        """
        Find a path between two nodes (simple BFS).

        Parameters
        ----------
        start_id : str
            ID of starting node.
        end_id : str
            ID of target node.

        Returns
        -------
        list of str
            Path of node IDs from start to end (inclusive).
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return []

        visited = set()
        queue = [(start_id, [start_id])]

        while queue:
            node_id, path = queue.pop(0)
            if node_id == end_id:
                return path

            if node_id not in visited:
                visited.add(node_id)
                for neighbor in self.adjacency.get(node_id, []):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

        return []  # No path found

    def to_dict(self) -> dict:
        """Export hypergraph to dictionary (for serialization)."""
        return {
            'name': self.name,
            'nodes': list(self.nodes.keys()),
            'handovers': list(self.handovers.keys()),
            'adjacency': self.adjacency,
            'global_coherence': self.global_coherence,
            'global_phi': self.global_phi,
            'history': self.history
        }

    def __repr__(self):
        return (f"ArkheHypergraph(name={self.name}, "
                f"nodes={len(self.nodes)}, "
                f"handovers={len(self.handovers)}, "
                f"C_global={self.global_coherence:.4f})")
