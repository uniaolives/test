"""
Visualization tools for Arkhe(n) quantum hypergraphs.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qutip import Bloch

def plot_hypergraph(hypergraph, figsize=(10, 8), node_size=500, edge_width=2):
    """
    Plot an ArkheHypergraph as a network graph.

    Parameters
    ----------
    hypergraph : ArkheHypergraph
        Hypergraph to visualize.
    figsize : tuple, default=(10,8)
        Figure size.
    node_size : int, default=500
        Size of nodes.
    edge_width : int, default=2
        Width of edges.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    G = nx.Graph()

    # Add nodes
    for node_id, node in hypergraph.nodes.items():
        G.add_node(node_id,
                   coherence=node.coherence,
                   winding=node.winding_number)

    # Add edges from handovers
    for handover_id, handover in hypergraph.handovers.items():
        src_id = handover.source.node_id
        tgt_id = handover.target.node_id
        G.add_edge(src_id, tgt_id,
                   handover_id=handover_id,
                   protocol=handover.protocol)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Node colors based on coherence
    coherences = [hypergraph.nodes[n].coherence for n in G.nodes]
    cmap = plt.cm.RdYlGn  # Red (low) to Green (high)

    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color=coherences,
                           cmap=cmap,
                           node_size=node_size,
                           ax=ax)

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                           width=edge_width,
                           alpha=0.5,
                           ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos,
                            font_size=8,
                            ax=ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(coherences)
    plt.colorbar(sm, ax=ax, label='Coherence C')

    ax.set_title(f"Arkhe Hypergraph: {hypergraph.name}\n"
                 f"C_global = {hypergraph.global_coherence:.4f}, "
                 f"Φ = {hypergraph.global_phi:.4f}")
    ax.axis('off')

    plt.tight_layout()
    return fig, ax


def plot_handover_timeline(hypergraph, figsize=(12, 5)):
    """
    Plot timeline of handovers with coherence evolution.

    Parameters
    ----------
    hypergraph : ArkheHypergraph
        Hypergraph with history.
    figsize : tuple, default=(12,5)
        Figure size.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if not hypergraph.history:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No history available", ha='center', va='center')
        return fig, ax

    history = hypergraph.history
    times = np.arange(len(history))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Coherence over time
    coherences = [h['global_coherence'] for h in history]
    ax1.plot(times, coherences, 'g-', linewidth=2)
    ax1.set_ylabel('Global Coherence C')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=hypergraph.global_coherence, color='r', linestyle='--',
                label=f'Final: {hypergraph.global_coherence:.4f}')
    ax1.legend()

    # Phi over time
    phis = [h['global_phi'] for h in history]
    ax2.plot(times, phis, 'b-', linewidth=2)
    ax2.set_xlabel('Handover count')
    ax2.set_ylabel('Integrated Info Φ')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=hypergraph.global_phi, color='r', linestyle='--',
                label=f'Final: {hypergraph.global_phi:.4f}')
    ax2.legend()

    fig.suptitle(f"Evolution of {hypergraph.name}")
    plt.tight_layout()
    return fig, (ax1, ax2)
