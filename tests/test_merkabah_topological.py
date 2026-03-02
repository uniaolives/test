# tests/test_merkabah_topological.py
import pytest
import torch
import numpy as np
from papercoder_kernel.merkabah.topological.anyon import AnyonLayer, TopologicallyProtectedFederation

def test_anyon_exchange_phase():
    nodes = ['Alpha', 'Beta', 'Gamma']
    anyon = AnyonLayer(nodes)

    # Initial state is identity
    assert torch.allclose(anyon.state_matrix, torch.eye(3, dtype=torch.complex64))

    # Exchange Alpha and Beta
    anyon.exchange('Alpha', 'Beta')

    # Check that it's no longer identity
    assert not torch.allclose(anyon.state_matrix, torch.eye(3, dtype=torch.complex64))

    # Check unitarity (U^H @ U = I)
    identity_approx = anyon.state_matrix.conj().T @ anyon.state_matrix
    assert torch.allclose(identity_approx, torch.eye(3, dtype=torch.complex64), atol=1e-6)

def test_braid_evolution_coherence():
    nodes = ['Alpha', 'Beta', 'Gamma', 'Self']
    anyon = AnyonLayer(nodes)

    sequence = [('Alpha', 'Beta'), ('Beta', 'Gamma'), ('Gamma', 'Self')]
    for a, b in sequence:
        anyon.exchange(a, b)

    result = anyon.braid_evolution(sequence)
    assert result['braid_length'] == 3
    assert 0 <= result['coherence'] <= 1.0

def test_topological_federation_logic():
    # Mock transport
    class MockTransport:
        def handover_quantum_state(self, target, data):
            return True

    nodes = ['Alpha', 'Beta', 'Gamma', 'Self']
    anyon = AnyonLayer(nodes)
    fed = TopologicallyProtectedFederation(MockTransport(), anyon)

    result = fed.execute_protected_logic("COMPUTE_PHAISTOS")
    assert result['braid_length'] == 3
    assert isinstance(result['final_topology'], torch.Tensor)

    # Verify that different instructions produce different topologies
    anyon2 = AnyonLayer(nodes)
    fed2 = TopologicallyProtectedFederation(MockTransport(), anyon2)
    result_stabilize = fed2.execute_protected_logic("STABILIZE")

    assert not torch.allclose(result['final_topology'], result_stabilize['final_topology'])
