#!/usr/bin/env python3
# asi/network/honeycomb_435.py
# Adjacency Graph for {4,3,5} Hyperbolic Honeycomb
# Block Ω+∞+171

import networkx as nx

class Honeycomb435:
    """Adjacency representation of the {4,3,5} hyperbolic honeycomb."""
    def __init__(self, layers=2):
        self.graph = nx.Graph()
        self.layers = layers
        self.generate()

    def generate(self):
        """
        Generate a finite subset of the {4,3,5} honeycomb.
        In {4,3,5}, each cell is a cube and has degree 4 in the adjacency graph.
        """
        # Start with a central cube
        self.graph.add_node(0, layer=0)

        current_layer_nodes = [0]
        next_id = 1

        for layer in range(1, self.layers + 1):
            next_layer_nodes = []
            for node in current_layer_nodes:
                # Each node should have 4 neighbors
                existing_degree = self.graph.degree(node)
                needed = 4 - existing_degree

                for _ in range(needed):
                    self.graph.add_node(next_id, layer=layer)
                    self.graph.add_edge(node, next_id)
                    next_layer_nodes.append(next_id)
                    next_id += 1
            current_layer_nodes = next_layer_nodes

    def get_adjacency_matrix(self):
        return nx.adjacency_matrix(self.graph).toarray()

    def get_laplacian(self):
        return nx.laplacian_matrix(self.graph).toarray()

    def get_summary(self):
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }

if __name__ == "__main__":
    h = Honeycomb435(layers=3)
    summary = h.get_summary()
    print(f"Honeycomb {4,3,5} (Finite Subset):")
    print(f"  Nodes: {summary['nodes']}")
    print(f"  Edges: {summary['edges']}")
    print(f"  Avg Degree: {summary['avg_degree']:.2f}")

    # Check degree distribution
    degrees = [d for n, d in h.graph.degree()]
    print(f"  Degree Distribution: {set(degrees)}")
