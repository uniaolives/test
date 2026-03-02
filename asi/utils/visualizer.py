import matplotlib.pyplot as plt
import networkx as nx
from core.hypergraph import Hypergraph

def show_graph(h: Hypergraph, dim=2):
    """Visualize the hypergraph as a graph (simplified: edges as connections between all node pairs)."""
    G = nx.Graph()
    for nid, node in h.nodes.items():
        G.add_node(nid, coherence=node.coherence)
    for edge in h.edges:
        # For visualization, connect all pairs in the hyperedge (simplification)
        nodes_list = list(edge.nodes)
        for i in range(len(nodes_list)):
            for j in range(i+1, len(nodes_list)):
                G.add_edge(nodes_list[i], nodes_list[j], weight=edge.weight)

    pos = nx.spring_layout(G, seed=42)
    colors = [G.nodes[n]['coherence'] for n in G.nodes]
    nx.draw(G, pos, node_color=colors, cmap=plt.cm.Blues, with_labels=True, font_size=8)
    plt.title(f"Hypergraph visualization (C_total={h.total_coherence():.3f})")
    plt.show()
