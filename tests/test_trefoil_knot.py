import pytest
from gateway.app.quantum.qiskit_circuits import trefoil_knot_circuit, QiskitInterface
from qiskit import QuantumCircuit

def test_trefoil_knot_circuit_generation():
    qc = trefoil_knot_circuit()
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 6
    assert qc.num_clbits == 6

@pytest.mark.asyncio
async def test_trefoil_knot_simulation():
    iface = QiskitInterface()
    qc = trefoil_knot_circuit()

    counts = iface.run_simulation(qc)
    assert isinstance(counts, dict)
    assert "error" not in counts

    # Check that we only have counts for 6-bit strings
    for bitstring in counts.keys():
        assert len(bitstring) == 6
