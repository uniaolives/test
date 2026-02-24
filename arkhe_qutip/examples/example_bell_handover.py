"""
Example: Creating a Bell state through quantum handovers.
"""

import numpy as np
from qutip import basis, tensor, sigmax, sigmay, sigmaz, about
from arkhe_qutip.core.arkhe_qobj import ArkheQobj
from arkhe_qutip.core.handover import QuantumHandover
from arkhe_qutip.core.hypergraph import ArkheHypergraph

def main():
    print("="*60)
    print("Arkhe(n)-QuTiP Example: Bell State via Handover")
    print("="*60)

    # Create two qubit nodes
    qubit0 = ArkheQobj(basis(2, 0), node_id="Qubit_0")
    qubit1 = ArkheQobj(basis(2, 0), node_id="Qubit_1")

    print(f"\nInitial nodes:")
    print(f"  {qubit0}")
    print(f"  {qubit1}")

    # Create hypergraph
    hg = ArkheHypergraph("Bell Experiment")
    hg.add_node(qubit0).add_node(qubit1)

    # Handover 1: Hadamard on qubit0
    from qutip.core.gates import hadamard_transform
    H = hadamard_transform(1)  # Hadamard on single qubit

    # Create handover (source acts on itself, affects target indirectly)
    h_hadamard = QuantumHandover(
        handover_id="H_0",
        source=qubit0,
        target=qubit0,  # self-handover for gate application
        operator=H,
        metadata={'gate': 'Hadamard'}
    )
    hg.add_handover(h_hadamard)

    # Execute Hadamard
    print("\nApplying Hadamard on qubit 0...")
    result = hg.execute_handover("H_0")
    print(f"  After handover: {qubit0}")

    # Handover 2: CNOT between qubits
    from qutip.core.gates import cnot
    CNOT = cnot()  # CNOT with control=0, target=1

    # CNOT acts on both qubits - need combined operator
    # For simplicity, we'll create a handover that affects both
    h_cnot = QuantumHandover(
        handover_id="CNOT",
        source=qubit0,
        target=qubit1,
        operator=CNOT,  # Actually acts on combined system
        metadata={'gate': 'CNOT'}
    )
    hg.add_handover(h_cnot)

    # Execute CNOT
    print("\nApplying CNOT between qubits...")
    result = hg.execute_handover("CNOT")

    # Final state should be Bell state (|00> + |11>)/√2
    bell_state = (tensor(basis(2,0), basis(2,0)) +
                  tensor(basis(2,1), basis(2,1))).unit()

    print(f"\nFinal hypergraph: {hg}")
    print(f"Global coherence: {hg.global_coherence:.4f}")
    print(f"Winding number: {hg.compute_winding_number()}")

    # Check if we have Bell state (simplified)
    print("\nBell state verification:")
    print("  Expected: (|00> + |11>)/√2")

    # In a real implementation, would need to compute overlap with Bell state
    # This requires extracting the combined state from the nodes

    print("\n✅ Example completed")

if __name__ == "__main__":
    main()
