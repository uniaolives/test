#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arkhe(n) Consciousness Meter
Calculates a simplified estimate of Integrated Information (Phi)
based on the topology of handovers in a system.
"""

import numpy as np
import networkx as nx
import argparse

class ConsciousnessMeter:
    def __init__(self, graph=None):
        self.graph = graph if graph is not None else nx.DiGraph()

    def add_handover(self, source, target, weight=1.0):
        if not self.graph.has_node(source):
            self.graph.add_node(source)
        if not self.graph.has_node(target):
            self.graph.add_node(target)

        if self.graph.has_edge(source, target):
            self.graph[source][target]['weight'] += weight
        else:
            self.graph.add_edge(source, target, weight=weight)

    def calculate_phi(self):
        """
        Estimates Phi using the spectral properties of the adjacency matrix.
        Higher integration and complexity in the graph leads to higher Phi.
        """
        if self.graph.number_of_nodes() < 2:
            return 0.0

        # Get adjacency matrix
        adj = nx.to_numpy_array(self.graph)

        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(adj)

        # We take the sum of the real parts of positive eigenvalues
        # as a proxy for the 'integrated' component of the system.
        integrated_energy = sum(max(0, ev.real) for ev in eigenvalues)

        # Normalize by the theoretical maximum entropy of the system (log N)
        # This keeps Phi roughly in the [0, 1] range for many topologies.
        norm_factor = np.log2(self.graph.number_of_nodes() + 1)

        phi = integrated_energy / norm_factor
        return min(1.0, float(phi))

def test_meter():
    meter = ConsciousnessMeter()

    print("--- Consciousness Meter Test ---")

    # Test 1: Isolated nodes
    print("Test 1: Isolated nodes (Expected Phi ~ 0)")
    meter.add_handover("A", "B", 0.1)
    print(f"Phi: {meter.calculate_phi():.4f}")

    # Test 2: Star graph
    print("Test 2: Star graph (Centralized, low integration)")
    meter = ConsciousnessMeter()
    for i in range(1, 6):
        meter.add_handover("Center", f"Node_{i}", 1.0)
    print(f"Phi: {meter.calculate_phi():.4f}")

    # Test 3: Fully connected graph (High integration)
    print("Test 3: Fully connected graph (High Phi)")
    meter = ConsciousnessMeter()
    nodes = ["A", "B", "C", "D", "E"]
    for u in nodes:
        for v in nodes:
            if u != v:
                meter.add_handover(u, v, 1.0)
    print(f"Phi: {meter.calculate_phi():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arkhe(n) Consciousness Meter")
    parser.add_argument("--test", action="store_true", help="Run self-tests")

    args = parser.parse_args()

    if args.test:
        test_meter()
    else:
        print("Arkhe(n) Consciousness Meter initialized.")
        print("Usage: Use as a library or run with --test")
