import json
import uuid
from typing import Dict, List, Set, Any, Optional

class Node:
    """A node in the hypergraph."""
    def __init__(self, node_id: Optional[str] = None, data: Optional[Dict] = None):
        self.id = node_id or str(uuid.uuid4())
        self.data = data or {}
        self.coherence = 1.0          # local coherence C_local

    def __repr__(self):
        return f"Node({self.id[:8]}... c={self.coherence:.3f})"

class Hyperedge:
    """An edge connecting a set of nodes."""
    def __init__(self, nodes: Set[str], weight: float = 1.0, edge_id: Optional[str] = None):
        self.id = edge_id or str(uuid.uuid4())
        self.nodes = nodes
        self.weight = weight

    def __repr__(self):
        return f"Edge({self.id[:8]}... nodes={len(self.nodes)}, w={self.weight:.3f})"

class Hypergraph:
    """The hypergraph H = (V, E, ∂)."""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Hyperedge] = []

    def add_node(self, node_id: Optional[str] = None, data: Optional[Dict] = None) -> Node:
        node = Node(node_id, data)
        self.nodes[node.id] = node
        return node

    def add_edge(self, node_ids: Set[str], weight: float = 1.0) -> Hyperedge:
        # Ensure all nodes exist
        for nid in node_ids:
            if nid not in self.nodes:
                raise ValueError(f"Node {nid} does not exist")
        edge = Hyperedge(node_ids, weight)
        self.edges.append(edge)
        return edge

    def bootstrap_step(self) -> None:
        """Single bootstrap iteration: update node coherence based on incident edges."""
        # For each node, coherence = average weight of incident edges
        for node in self.nodes.values():
            incident_weights = [e.weight for e in self.edges if node.id in e.nodes]
            if incident_weights:
                node.coherence = sum(incident_weights) / len(incident_weights)
            else:
                node.coherence = 0.0

    def total_coherence(self) -> float:
        """C_total = average of all node coherences."""
        if not self.nodes:
            return 0.0
        return sum(n.coherence for n in self.nodes.values()) / len(self.nodes)

    def to_json(self) -> dict:
        return {
            "nodes": {nid: node.data for nid, node in self.nodes.items()},
            "edges": [{"nodes": list(e.nodes), "weight": e.weight} for e in self.edges]
        }

    @classmethod
    def from_json(cls, data: dict) -> "Hypergraph":
        h = cls()
        for nid, ndata in data["nodes"].items():
            h.add_node(nid, ndata)
        for e in data["edges"]:
            h.add_edge(set(e["nodes"]), e["weight"])
        return h
