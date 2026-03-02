#!/usr/bin/env python3
"""
6G Semantic Communication Simulator
Based on the SixGARROW.anl specification and the ANL core.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metalanguage.anl import Node, Handover, Hypergraph, Protocol

# ======================================================
# 1. SPECIALIZED NODES
# ======================================================

class SixG_UE(Node):
    def __init__(self, node_id, battery=1.0, data_rate=100.0):
        super().__init__(node_id, None, {})
        self.battery = battery
        self.tx_power = 1.0 # Watts
        self.data_rate = data_rate # Mbps

    def transmit_energy(self, size_mbits):
        time_sec = size_mbits / self.data_rate
        return self.tx_power * time_sec

    def compute_energy(self, flops):
        return flops * 1e-9 # 1 nJ/FLOP

class SemanticPlane(Node):
    def __init__(self, node_id, embedding_dim=64):
        super().__init__(node_id, None, {})
        self.embedding_dim = embedding_dim
        self.pca = None

    def fit(self, data):
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.embedding_dim)
        self.pca.fit(data)

    def encode(self, data):
        return self.pca.transform(data.reshape(1, -1))[0] if self.pca else data[:self.embedding_dim]

    def decode(self, emb):
        return self.pca.inverse_transform(emb.reshape(1, -1))[0] if self.pca else emb

# ======================================================
# 2. SIMULATION
# ======================================================

def run_sim():
    print("--- 6G Semantic Communication Simulation ---")
    ue = SixG_UE("UE_01")
    ran = Node("RAN_01", None, {})
    sp = SemanticPlane("SemanticPlane")

    # Fake data: 100 samples of 1000-dim vectors
    data_samples = np.random.randn(100, 1000)
    sp.fit(data_samples)

    results = []

    for i in range(50):
        sample = data_samples[i]

        # 1. Classic Transmission (Bit-stream)
        bits_classic = 1000 * 32 # float32
        energy_classic = ue.transmit_energy(bits_classic / 1e6)

        # 2. Semantic Transmission
        emb = sp.encode(sample)
        bits_semantic = len(emb) * 32
        energy_tx_sem = ue.transmit_energy(bits_semantic / 1e6)
        energy_comp_sem = ue.compute_energy(1000 * 10) # encoding cost
        energy_semantic = energy_tx_sem + energy_comp_sem

        # Fidelity
        reconstructed = sp.decode(emb)
        mse = np.mean((sample - reconstructed)**2)

        ue.battery -= energy_semantic

        results.append({
            'step': i,
            'energy_classic': energy_classic,
            'energy_semantic': energy_semantic,
            'mse': mse,
            'compression': bits_classic / bits_semantic
        })

        if ue.battery < 0.05: break

    # Summary
    avg_comp = np.mean([r['compression'] for r in results])
    avg_saving = np.mean([r['energy_classic']/r['energy_semantic'] for r in results])
    print(f"Avg Compression: {avg_comp:.2f}x")
    print(f"Avg Energy Saving: {avg_saving:.2f}x")
    print(f"Avg MSE: {np.mean([r['mse'] for r in results]):.4f}")

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot([r['energy_classic'] for r in results], label='Classic')
    plt.plot([r['energy_semantic'] for r in results], label='Semantic')
    plt.title("Energy Consumption (J)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([r['mse'] for r in results])
    plt.title("Semantic Reconstruction Error (MSE)")

    plt.tight_layout()
    plt.savefig('6g_semantic_results.png')
    print("Results saved to 6g_semantic_results.png")

if __name__ == "__main__":
    run_sim()
