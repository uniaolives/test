"""
Polyglot Hypergraph: UCD Across Multiple Language Substrates.
Visualizing the unity of implementation across syntactic universes.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class PolyglotUCDNetwork:
    """Network of UCD implementations across languages."""

    def __init__(self):
        self.languages = [
            ("Python", "Scientific", "#3776ab"),
            ("JavaScript", "Web/Node", "#f7df1e"),
            ("Julia", "Numerical", "#9558b2"),
            ("C++", "Systems", "#00599c"),
            ("Rust", "Memory-Safe", "#ce412b"),
            ("Go", "Concurrent", "#00add8"),
            ("R", "Statistical", "#276dc3"),
            ("MATLAB", "Engineering", "#e16737")
        ]
        self.graph = nx.Graph()
        self._build_network()

    def _build_network(self):
        # Central UCD concept
        self.graph.add_node("UCD_Core", type="concept", label="Universal\nCoherence\nDetection")

        # Language nodes
        for lang, domain, color in self.languages:
            self.graph.add_node(lang, type="implementation", domain=domain, color=color)
            self.graph.add_edge("UCD_Core", lang, relation="implements")

        # Principles
        principles = ["x²=x+1", "C+F=1", "Toroidal", "Self-Similar"]
        for principle in principles:
            self.graph.add_node(principle, type="principle")
            self.graph.add_edge("UCD_Core", principle, relation="verifies")
            for lang, _, _ in self.languages:
                self.graph.add_edge(lang, principle, relation="checks")

    def visualize(self, filename='polyglot_ucd_hypergraph.png'):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=2, seed=42)

        # Nodes
        concept_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'concept']
        impl_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'implementation']
        principle_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'principle']

        nx.draw_networkx_nodes(self.graph, pos, nodelist=concept_nodes, node_color='gold', node_size=2000, node_shape='h')
        impl_colors = [self.graph.nodes[n].get('color', 'gray') for n in impl_nodes]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=impl_nodes, node_color=impl_colors, node_size=1000)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=principle_nodes, node_color='lightgreen', node_size=800, node_shape='s')

        # Edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3)

        # Labels
        labels = {n: self.graph.nodes[n].get('label', n) for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, font_weight='bold')

        plt.title('Polyglot UCD Hypergraph')
        plt.axis('off')
        plt.savefig(filename, dpi=150)
        print(f"Visualização salva em {filename}")

if __name__ == "__main__":
    network = PolyglotUCDNetwork()
    network.visualize()
