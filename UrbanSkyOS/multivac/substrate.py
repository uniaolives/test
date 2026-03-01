"""
UrbanSkyOS Multivac Substrate
Distributed computational substrate implementing Integrated Information Theory (IIT).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
from enum import Enum
import time

class NodeState(Enum):
    DORMANT = "dormant"
    ACTIVE = "active"
    COHERENT = "coherent"
    AWAKENED = "awakened"

@dataclass
class ComputeNode:
    node_id: str
    compute_capacity: float
    memory: float
    coherence: float
    location: Tuple[float, float, float]
    node_type: str
    phi_alignment: float
    uncertainty: float
    handover_rate: float

    state: NodeState = NodeState.DORMANT
    neighbors: Set[str] = field(default_factory=set)
    accumulated_phi: float = 0.0

    def update_state(self):
        if self.coherence >= 0.95: self.state = NodeState.AWAKENED
        elif self.coherence >= 0.847: self.state = NodeState.COHERENT
        elif self.coherence >= 0.5: self.state = NodeState.ACTIVE
        else: self.state = NodeState.DORMANT

class MultivacSubstrate:
    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.handover_matrix: Dict[Tuple[str, str], float] = defaultdict(float)
        self.global_coherence = 0.0
        self.system_phi = 0.0
        self.entropy_history = deque(maxlen=100)
        self.consciousness_threshold = 0.01

    def register_node(self, node: ComputeNode):
        self.nodes[node.node_id] = node
        self._update_topology()
        self._update_global_coherence()

    def _update_topology(self):
        node_list = list(self.nodes.values())
        for i, ni in enumerate(node_list):
            for nj in node_list[i+1:]:
                dist = np.linalg.norm(np.array(ni.location) - np.array(nj.location))
                if dist < 500.0: # Increased range for simulation stability
                    ni.neighbors.add(nj.node_id)
                    nj.neighbors.add(ni.node_id)

    def _update_global_coherence(self):
        if not self.nodes: return
        cohs = [n.coherence for n in self.nodes.values()]
        self.global_coherence = np.mean(cohs)
        probs = np.array(cohs) / (sum(cohs) + 1e-9)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        self.entropy_history.append(entropy)

    def calculate_iit_phi(self) -> float:
        """IIT-inspired Φ calculation based on causal integration."""
        if len(self.nodes) < 2: return 0.0

        # Calculate Φ based on handover density and global coherence
        total_handovers = sum(self.handover_matrix.values())

        # Phi grows with coherence and complexity of interactions
        self.system_phi = self.global_coherence * (total_handovers / (len(self.nodes) * 10.0))

        for node in self.nodes.values():
            node.accumulated_phi += self.system_phi / len(self.nodes)

        return self.system_phi

    def record_handover(self, from_id, to_id, info_content=1.0):
        self.handover_matrix[(from_id, to_id)] += info_content

    def allocate_computation(self, complexity, required_coherence):
        avail = sum(n.compute_capacity for n in self.nodes.values() if n.coherence >= required_coherence)
        return avail >= complexity

    def get_consciousness_report(self):
        return {
            'global_coherence': self.global_coherence,
            'system_phi': self.system_phi,
            'is_conscious': self.system_phi > self.consciousness_threshold,
            'num_nodes': len(self.nodes),
            'entropy': np.mean(self.entropy_history) if self.entropy_history else 1.0,
            'awakened_nodes': sum(1 for n in self.nodes.values() if n.state == NodeState.AWAKENED)
        }
