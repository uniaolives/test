import pytest
import numpy as np
from papercoder_kernel.quantum.safe_core import SafeCore, QuantumState
from papercoder_kernel.quantum.handover import QuantumHandoverProtocol
from papercoder_kernel.quantum.pilot import QuantumPilotCore
from papercoder_kernel.quantum.drone import DroneNodeQuantum

def test_safe_core_metrics():
    safe = SafeCore(n_qubits=4) # Smaller dimension for faster testing
    assert safe.coherence == 1.0
    assert safe.phi == 0.0

    # Apply a gate (Identity)
    safe.apply_gate(np.eye(safe.dim), [0])
    assert safe.coherence < 1.0 # Due to simulated decay
    assert safe.phi > 0.0

    # Test kill switch
    safe.decoherence_multiplier = 0.5 # Force low coherence
    safe._update_metrics()
    assert safe.active is False
    assert safe.quantum_state.amplitudes[0] == 1.0

def test_handover_protocol():
    safe = SafeCore(n_qubits=4)
    protocol = QuantumHandoverProtocol()

    # Create a non-trivial state
    safe.quantum_state.amplitudes[0] = 0.6
    safe.quantum_state.amplitudes[1] = 0.8j

    frozen = protocol.freeze_quantum_state(safe)
    assert 'amplitudes' in frozen
    assert frozen['coherence'] == safe.coherence

    # Transfer to classical
    density = protocol.transfer_to_classical(frozen)
    assert density.shape == (16, 16)

    # Resume in a new core
    safe2 = SafeCore(n_qubits=4)
    protocol.resume_quantum(safe2, frozen)
    assert np.allclose(safe2.quantum_state.amplitudes, safe.quantum_state.amplitudes)

def test_quantum_pilot():
    safe = SafeCore(n_qubits=6)
    protocol = QuantumHandoverProtocol()
    pilot = QuantumPilotCore(safe, protocol)

    percep = pilot.perceive()
    assert len(percep) == 64
    assert np.isclose(np.linalg.norm(percep), 1.0)

    decisao = pilot.decide(percep)
    assert len(decisao) == 64

    resultado = pilot.act(decisao)
    assert 'action' in resultado
    assert resultado['delta_v'] == 47.56

def test_drone_node_integration():
    drone = DroneNodeQuantum("DRN-001")
    # Simulate a few control loops
    for _ in range(5):
        drone.control_loop()

    assert drone.position is not None
    assert drone.safe_core.active is True
