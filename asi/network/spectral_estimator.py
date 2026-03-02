#!/usr/bin/env python3
# asi/network/spectral_estimator.py
# Distributed Spectral-Gap Estimator for Arkhe Protocol
# Block Ω+∞+171

import numpy as np

class DistributedSpectralEstimator:
    def __init__(self, num_nodes, adjacency):
        self.num_nodes = num_nodes
        self.adjacency = adjacency
        # Each node has its own component of the eigenvector estimate
        self.states = np.random.randn(num_nodes)
        self.states -= np.mean(self.states) # Orthogonal to constant vector
        self.states /= np.linalg.norm(self.states)

    def iterate(self):
        """One round of distributed gossip-based power iteration."""
        # Laplacian Lx = deg(x)*x - sum(neighbors)
        new_states = np.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            neighbors = np.where(self.adjacency[i] > 0)[0]
            degree = len(neighbors)
            local_sum = np.sum(self.states[neighbors])

            # Inverse iteration step: x_new = x - 0.5 * Lx
            laplacian_val = degree * self.states[i] - local_sum
            new_states[i] = self.states[i] - 0.1 * laplacian_val # Small step size

        # Global normalization (simulated via global consensus)
        new_states -= np.mean(new_states)
        norm = np.linalg.norm(new_states)
        if norm > 0:
            new_states /= norm

        self.states = new_states

    def estimate_lambda2(self) -> float:
        """Estimate the second smallest eigenvalue (Rayleigh quotient)."""
        lx = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            neighbors = np.where(self.adjacency[i] > 0)[0]
            degree = len(neighbors)
            local_sum = np.sum(self.states[neighbors])
            lx[i] = degree * self.states[i] - local_sum

        return np.dot(self.states, lx) / np.dot(self.states, self.states)

def run_simulation():
    # Simple line graph for testing
    num_nodes = 10
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        adj[i, i+1] = adj[i+1, i] = 1

    estimator = DistributedSpectralEstimator(num_nodes, adj)

    print(f"Starting Spectral Estimation (N={num_nodes})...")
    for i in range(50):
        estimator.iterate()
        if i % 10 == 0:
            l2 = estimator.estimate_lambda2()
            print(f"  Iteration {i}: Estimated λ2 = {l2:.4f}")

    final_l2 = estimator.estimate_lambda2()
    # Theoretical λ2 for line graph: 2 * (1 - cos(pi/n))
    theoretical = 2 * (1 - np.cos(np.pi / num_nodes))

    print(f"Final Estimated λ2: {final_l2:.4f}")
    print(f"Theoretical λ2: {theoretical:.4f}")
    return final_l2

if __name__ == "__main__":
    run_simulation()
