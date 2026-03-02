from .hypergraph import Hypergraph, Node

def coherence_total(h: Hypergraph) -> float:
    return h.total_coherence()

def coherence_local(h: Hypergraph, node_id: str) -> float:
    return h.nodes.get(node_id, Node()).coherence
