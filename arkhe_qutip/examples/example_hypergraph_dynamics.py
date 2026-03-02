"""
Example: Time evolution of a quantum hypergraph with ArkheSolver.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmaz, destroy, num
from arkhe_qutip.core.arkhe_qobj import ArkheQobj
from arkhe_qutip.core.hypergraph import ArkheHypergraph
from arkhe_qutip.dynamics.solver import ArkheSolver
from arkhe_qutip.visualization.hypergraph_plot import plot_handover_timeline

def main():
    print("="*60)
    print("Arkhe(n)-QuTiP Example: Hypergraph Dynamics")
    print("="*60)

    # Create nodes (qubits)
    nodes = []
    for i in range(3):
        node = ArkheQobj(basis(2, 1), node_id=f"Qubit_{i}")  # excited state
        nodes.append(node)

    # Hypergraph
    hg = ArkheHypergraph("Dynamics Test")
    for node in nodes:
        hg.add_node(node)

    print(f"\nInitial hypergraph: {hg}")

    # Create handovers (simulate interactions)
    from arkhe_qutip.core.handover import QuantumHandover

    for i in range(len(nodes)-1):
        h = QuantumHandover(
            handover_id=f"H_{i}_{i+1}",
            source=nodes[i],
            target=nodes[i+1],
            metadata={'type': 'interaction'}
        )
        hg.add_handover(h)

    # Simulate evolution with ArkheSolver
    # For simplicity, we'll just update coherence manually

    print("\nSimulating handover sequence...")
    for step in range(10):
        # Execute a handover
        handover_id = f"H_{step % 2}_{(step % 2)+1}"
        if handover_id in hg.handovers:
            hg.execute_handover(handover_id)
            print(f"  Step {step}: {handover_id} executed")

    print(f"\nFinal hypergraph: {hg}")
    print(f"Global coherence: {hg.global_coherence:.4f}")
    print(f"Global Φ: {hg.global_phi:.4f}")
    print(f"Winding number: {hg.compute_winding_number()}")

    # Plot timeline
    fig, axes = plot_handover_timeline(hg)
    plt.savefig("hypergraph_timeline.png")
    # plt.show() # Disabled for headless environment

    print("\n✅ Example completed. Timeline saved as hypergraph_timeline.png")

if __name__ == "__main__":
    main()
