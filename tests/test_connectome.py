# tests/test_connectome.py
import pytest
import torch
import numpy as np
from papercoder_kernel.core.biology.connectome import BiologicalHypergraph, ConnectomeStats

def test_connectome_stats():
    stats = ConnectomeStats(neurons=100, synapses=1000)
    assert stats.density == 10.0

def test_phi_evolution():
    brain = BiologicalHypergraph()
    initial_phi = brain.phi.item()

    # Step with zero noise should grow by alpha
    # phi_new = phi_old * exp(alpha * dt)
    # alpha = 0.01, dt = 1.0 -> exp(0.01) approx 1.01005
    brain.step(dt=1.0, external_noise=0.0)
    assert brain.phi.item() > initial_phi
    assert pytest.approx(brain.phi.item(), rel=1e-4) == initial_phi * np.exp(0.01)

def test_antifragility():
    # In antifragile regime, system should be able to recover or grow from noise
    brain = BiologicalHypergraph()
    phis = []
    for _ in range(100):
        phis.append(brain.step(dt=0.01))

    # Check if Phi remains bounded and positive
    assert all(p > 0 for p in phis)

def test_sonoluminescence():
    brain = BiologicalHypergraph()
    burst = brain.sonoluminescence_burst()
    # Signal (1e-15) / kT (4e-21) = 2.5e5
    assert pytest.approx(burst, rel=1e-2) == 2.5e5

def test_topology_report():
    brain = BiologicalHypergraph()
    report = brain.get_topology_report()
    assert report['nodes'] == 57000
    assert report['regime'] == 'antifragile'
