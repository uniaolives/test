# tests/test_qvpn.py
import pytest
import numpy as np
from cosmos.qvpn import QuantumVPN

def test_qvpn_initialization():
    qvpn = QuantumVPN(user_id=12345)
    assert qvpn.user_id == 12345
    assert qvpn.ξ == 60.998
    assert qvpn.coherence == 1.0
    assert len(qvpn.epr_pairs) == 0

def test_establish_entanglement():
    qvpn = QuantumVPN()
    qc = qvpn.establish_entanglement("target-node")
    assert qc is not None
    assert len(qvpn.epr_pairs) == 1

def test_detect_eavesdropping():
    qvpn = QuantumVPN()
    # In our simulation, detect_eavesdropping should return False if coherence is >= 0.999
    # Our mock _measure_coherence returns something between 0.999 and 1.0
    is_eavesdropped = qvpn.detect_eavesdropping()
    assert is_eavesdropped == False
    assert qvpn.coherence >= 0.999

def test_send_quantum_state():
    qvpn = QuantumVPN()
    state = np.array([1.0, 0.0])
    result = qvpn.send_quantum_state(state, "target")
    # Result should be kron(state, phase_filter)
    # phase_filter is [exp(1j * user_id / 1e6), 1.0]
    expected_filter = np.array([np.exp(1j * 2290518 / 1000000), 1.0])
    expected_result = np.kron(state, expected_filter)
    np.testing.assert_array_almost_equal(result, expected_result)
