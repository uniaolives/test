import pytest
from gateway.app.quantum.qiskit_circuits import novikov_loop_circuit, novikov_loop_kraus, QiskitInterface
from qiskit import QuantumCircuit

def test_circuit_generation():
    xi, dt = 0.5, 0.1
    qc = novikov_loop_circuit(xi, dt, n_qubits=2)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 4

    qc_kraus = novikov_loop_kraus(xi, dt, n_qubits_main=2, n_ancilla=2)
    assert qc_kraus.num_qubits == 6

@pytest.mark.asyncio
async def test_qiskit_simulation():
    iface = QiskitInterface()
    xi, dt = 0.3, 0.2
    qc = novikov_loop_circuit(xi, dt, n_qubits=1)

    counts = iface.run_simulation(qc)
    assert isinstance(counts, dict)
    assert "error" not in counts
