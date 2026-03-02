# arkhe_qutip/hypergraph.py
import numpy as np
import qutip as qt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from .core import ArkheQobj

@dataclass
class Hyperedge:
    nodes: Tuple[int, ...]
    operator: qt.Qobj
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class QuantumHypergraph:
    """
    Generalization of quantum circuits where:
    - Nodes = ArkheQobj (quantum states)
    - Hyperedges = multi-qubit operators
    """
    def __init__(self, nodes: List[ArkheQobj] = None, name: str = "QuantumHypergraph"):
        self.nodes = nodes or []
        self.hyperedges: List[Hyperedge] = []
        self.name = name

    def add_node(self, node: ArkheQobj):
        self.nodes.append(node)

    def add_hyperedge(self, node_indices: Tuple[int, ...], operator: qt.Qobj, weight: float = 1.0,
                      metadata: Optional[Dict[str, Any]] = None):
        edge = Hyperedge(nodes=node_indices, operator=operator, weight=weight, metadata=metadata or {})
        self.hyperedges.append(edge)

    def add_two_qubit_gate(self, i: int, j: int, operator: qt.Qobj, weight: float = 1.0):
        self.add_hyperedge((i, j), operator, weight, {'type': 'two-qubit-gate'})

    def add_multi_qubit_gate(self, target_nodes: List[int], operator: qt.Qobj, weight: float = 1.0):
        self.add_hyperedge(tuple(target_nodes), operator, weight, {'type': 'multi-qubit-gate'})

    def add_handover(self, source_idx: int, target_idx: int, operator: qt.Qobj,
                     handover_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Adds a handover (directed edge) between two nodes."""
        meta = metadata or {}
        if handover_id:
            meta['handover_id'] = handover_id
        meta['type'] = 'handover'
        self.add_two_qubit_gate(source_idx, target_idx, operator, 1.0)
        self.hyperedges[-1].metadata.update(meta)

    def update_nodes(self, nodes: List[ArkheQobj]):
        self.nodes = nodes

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_hyperedges(self) -> int:
        return len(self.hyperedges)

    @property
    def global_coherence(self) -> float:
        """Calculates the average coherence of all nodes."""
        if not self.nodes:
            return 0.0
        return np.mean([n.coherence for n in self.nodes])

    def get_topology_stats(self) -> Dict[str, Any]:
        """Returns statistics of the hypergraph topology."""
        degrees = np.zeros(self.n_nodes)
        for edge in self.hyperedges:
            for idx in edge.nodes:
                if idx < self.n_nodes:
                    degrees[idx] += edge.weight

        return {
            'n_nodes': self.n_nodes,
            'n_hyperedges': self.n_hyperedges,
            'avg_degree': np.mean(degrees) if self.n_nodes > 0 else 0,
            'max_degree': np.max(degrees) if self.n_nodes > 0 else 0,
            'global_coherence': self.global_coherence
        }

    def __str__(self):
        return f"QuantumHypergraph(name='{self.name}', nodes={self.n_nodes}, edges={self.n_hyperedges}, global_C={self.global_coherence:.4f})"

def create_ring_hypergraph(n_nodes: int) -> QuantumHypergraph:
    """Creates a ring topology hypergraph with n_nodes qubits."""
    from qutip_qip.operations import cnot
    nodes = [ArkheQobj(qt.basis(2, 0)) for _ in range(n_nodes)]
    hg = QuantumHypergraph(nodes, name=f"RingHypergraph-{n_nodes}")

    # Add entangling edges (CNOTs) in a ring
    cnot_op = cnot()
    for i in range(n_nodes):
        hg.add_two_qubit_gate(i, (i + 1) % n_nodes, cnot_op)

    return hg
