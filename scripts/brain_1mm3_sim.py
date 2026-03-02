# scripts/brain_1mm3_sim.py
import sys
import os
import time

# Add src to path
sys.path.append(os.path.abspath("src"))

from papercoder_kernel.core.biology.connectome import BiologicalHypergraph, ConnectomeStats

def main():
    print("--- Biological Hypergraph Simulation (1 mm3 Connectome) ---")
    stats = ConnectomeStats()
    brain = BiologicalHypergraph(stats)

    print(f"Nodes: {stats.neurons}")
    print(f"Edges: {stats.synapses}")
    print(f"Synaptic Density: {stats.density:.2f}")

    print("\nRunning evolution loop (Antifragile Regime)...")
    for i in range(10):
        phi = brain.step(dt=0.1)
        burst = brain.sonoluminescence_burst()
        print(f"Step {i}: Phi = {phi:.4f}, Sonoluminescence Burst = {burst:.2e}")
        time.sleep(0.1)

    print("\nFinal Topology Report:")
    report = brain.get_topology_report()
    for k, v in report.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
