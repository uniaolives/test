# arkhe_qutip/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Any, List, Optional, Tuple

def plot_hypergraph(hg: Any, layout: str = 'spring', ax: Optional[plt.Axes] = None):
    """Plots the quantum hypergraph topology."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    G = nx.Graph()
    for i, node in enumerate(hg.nodes):
        G.add_node(i, coherence=node.coherence)

    for edge in hg.hyperedges:
        if len(edge.nodes) == 2:
            G.add_edge(edge.nodes[0], edge.nodes[1], weight=edge.weight)
        else:
            # For multi-node edges, add all pairs to visualize connectivity
            from itertools import combinations
            for u, v in combinations(edge.nodes, 2):
                G.add_edge(u, v, weight=edge.weight)

    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G)

    node_colors = [G.nodes[n]['coherence'] for n in G.nodes]

    nx.draw(G, pos, ax=ax, with_labels=True,
            node_color=node_colors, cmap=plt.cm.viridis,
            node_size=500, font_color='white')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Coherence")

    ax.set_title(f"Quantum Hypergraph: {hg.name}")
    return plt.gcf(), ax

def plot_coherence_trajectory(trajectory: List[float], events: Optional[List[Any]] = None,
                             ax: Optional[plt.Axes] = None):
    """Plots the evolution of coherence over time or steps."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    steps = np.arange(len(trajectory))
    ax.plot(steps, trajectory, marker='o', linestyle='-', color='blue', label="Coherence")

    if events:
        # If we have events, we can mark them
        # Note: events indices might not match trajectory if it's high-res
        pass

    ax.set_xlabel("Step / Event Index")
    ax.set_ylabel("Coherence (Purity)")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_title("Coherence Trajectory")
    ax.legend()

    return plt.gcf(), ax
