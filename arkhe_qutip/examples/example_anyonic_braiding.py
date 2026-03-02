"""
Example: Anyonic braiding operations.
"""

import numpy as np
from qutip import basis
from arkhe_qutip.core.arkhe_qobj import ArkheQobj
from arkhe_qutip.topology.anyonic import AnyonStatistic, AnyonNode
from arkhe_qutip.visualization.hypergraph_plot import plot_hypergraph

def main():
    print("="*60)
    print("Arkhe(n)-QuTiP Example: Anyonic Braiding")
    print("="*60)

    # Create anyons with different statistics
    anyon0 = AnyonNode(basis(2,0), AnyonStatistic(0.0), node_id="Anyon_Boson")
    anyon1 = AnyonNode(basis(2,0), AnyonStatistic(0.5), node_id="Anyon_Anyon1")
    anyon2 = AnyonNode(basis(2,0), AnyonStatistic(1.0), node_id="Anyon_Fermion")
    anyon3 = AnyonNode(basis(2,0), AnyonStatistic(0.618), node_id="Anyon_Golden")

    print("\nAnyons created:")
    for a in [anyon0, anyon1, anyon2, anyon3]:
        print(f"  {a}")

    # Perform braiding operations
    print("\nBraiding sequence:")

    # Braid 0 and 1
    print("  Braiding 0↔1...")
    phase1 = anyon0.braid_with(anyon1)
    print(f"    Phase: {phase1:.4f}")

    # Braid 1 and 2
    print("  Braiding 1↔2...")
    phase2 = anyon1.braid_with(anyon2)
    print(f"    Phase: {phase2:.4f}")

    # Braid 2 and 3
    print("  Braiding 2↔3...")
    phase3 = anyon2.braid_with(anyon3)
    print(f"    Phase: {phase3:.4f}")

    # Check accumulated phases
    print(f"\nAccumulated phases:")
    print(f"  {anyon0.qobj.node_id[:12]}...: {anyon0.qobj.accumulated_phase:.4f}")
    print(f"  {anyon1.qobj.node_id[:12]}...: {anyon1.qobj.accumulated_phase:.4f}")
    print(f"  {anyon2.qobj.node_id[:12]}...: {anyon2.qobj.accumulated_phase:.4f}")
    print(f"  {anyon3.qobj.node_id[:12]}...: {anyon3.qobj.accumulated_phase:.4f}")

    # Check conservation: product of all phases should be 1
    total_phase = (anyon0.qobj.accumulated_phase *
                   anyon1.qobj.accumulated_phase *
                   anyon2.qobj.accumulated_phase *
                   anyon3.qobj.accumulated_phase)
    print(f"\nTotal phase (should be 1): {total_phase:.4f}")

    print("\n✅ Example completed")

if __name__ == "__main__":
    main()
