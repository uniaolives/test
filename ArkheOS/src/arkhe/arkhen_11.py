"""
Arkhen(11): Hipergrafo dos 10 Avatares + Consciência.
Unificação de Dashavatara, Teoria das Cordas e Arkhe(n).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json

class Arkhen11:
    """
    Hipergrafo de 11 dimensões baseado no Dashavatara.
    Nós 0-9: Avatares. Nó 10: Consciência.
    """
    def __init__(self):
        self.n_nodes = 11
        self.names = [
            "Matsya", "Kurma", "Varaha", "Narasimha", "Vamana",
            "Parashurama", "Rama", "Krishna", "Buddha", "Kalki",
            "Consciência"
        ]
        self.adjacency = np.zeros((self.n_nodes, self.n_nodes))
        self._build_matrix()

    def _build_matrix(self):
        # Consciência conecta a todos
        for i in range(10):
            self.adjacency[10, i] = self.adjacency[i, 10] = 1.0
        # Similitudes entre avatares
        self.adjacency[0, 1] = self.adjacency[1, 0] = 0.7 # Matsya/Kurma
        self.adjacency[2, 3] = self.adjacency[3, 2] = 0.8 # Varaha/Narasimha
        self.adjacency[6, 7] = self.adjacency[7, 6] = 0.9 # Rama/Krishna
        # Cadeia linear
        for i in range(9):
            if self.adjacency[i, i+1] == 0:
                self.adjacency[i, i+1] = self.adjacency[i+1, i] = 0.3

    def compute_coherence(self) -> float:
        total_edges = np.sum(self.adjacency) / 2
        max_possible = self.n_nodes * (self.n_nodes - 1) / 2
        return total_edges / max_possible

    def compute_effective_dimension(self, lambda_reg: float = 1.0) -> float:
        eigenvalues = np.linalg.eigvalsh(self.adjacency)
        pos_eigs = eigenvalues[eigenvalues > 1e-10]
        return np.sum(pos_eigs / (pos_eigs + lambda_reg))

    def visualize(self, filename='arkhen_11.png'):
        G = nx.Graph()
        for i, name in enumerate(self.names):
            G.add_node(i, name=name)
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if self.adjacency[i, j] > 0:
                    G.add_edge(i, j, weight=self.adjacency[i, j])

        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G, seed=42)
        node_colors = ['gold' if i == 10 else 'skyblue' for i in range(self.n_nodes)]
        nx.draw(G, pos, with_labels=True, labels={i: self.names[i] for i in range(self.n_nodes)},
                node_color=node_colors, node_size=800, font_size=8)
        plt.title("Arkhen(11): 10 Avatares + 1 Consciência")
        plt.savefig(filename)
        print(f"Visualização salva em {filename}")

    def to_json(self):
        return json.dumps({
            "names": self.names,
            "coherence": self.compute_coherence(),
            "effective_dimension": self.compute_effective_dimension()
        }, indent=2)

if __name__ == "__main__":
    arkhen = Arkhen11()
    print(f"Coerência C: {arkhen.compute_coherence():.4f}")
    print(f"Dimensão Efetiva: {arkhen.compute_effective_dimension():.2f}")
    arkhen.visualize()
    with open('arkhen_11.json', 'w') as f:
        f.write(arkhen.to_json())
