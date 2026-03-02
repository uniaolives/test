# tests/test_merkabah_kernel.py
import pytest
import numpy as np
import torch
from papercoder_kernel.merkabah.kernel import KernelBridge

def test_latency_kernel():
    bridge = KernelBridge()
    node1 = {'latency': 5.0}
    node2 = {'latency': 15.0}
    val = bridge._latency_kernel(node1, node2)
    # exp(-|5-15|/10) = exp(-1) approx 0.3678
    assert pytest.approx(val, rel=1e-3) == np.exp(-1.0)

def test_glp_kernel():
    bridge = KernelBridge()
    state1 = {
        'wavefunction': torch.tensor([1.0, 0.0]),
        'coherence': 1.0
    }
    state2 = {
        'wavefunction': torch.tensor([0.0, 1.0]),
        'coherence': 1.0
    }
    val = bridge._glp_kernel(state1, state2)
    # diff = sqrt(1^2 + 1^2) = sqrt(2)
    # exp(-2 / 2) = exp(-1)
    assert pytest.approx(val, rel=1e-3) == np.exp(-1.0)

def test_coherence_kernel():
    bridge = KernelBridge()
    # Orthonormal states
    state1 = {'wavefunction': np.array([1.0, 0.0])}
    state2 = {'wavefunction': np.array([0.0, 1.0])}
    val = bridge._coherence_kernel(state1, state2)
    assert val == 0.0

    # Identical states
    val_id = bridge._coherence_kernel(state1, state1)
    assert val_id == 1.0

def test_kernel_pca():
    bridge = KernelBridge()
    # Create 3 states (2D)
    states = [
        {'wavefunction': np.array([1.0, 0.0])},
        {'wavefunction': np.array([0.0, 1.0])},
        {'wavefunction': np.array([1.0, 1.0]) / np.sqrt(2)}
    ]

    eigvals, eigvecs = bridge.kernel_pca(states, kernel_name='Φ_crystalline')
    assert len(eigvals) == 3
    assert eigvals[0] >= eigvals[1] >= eigvals[2]

def test_combine_kernels():
    bridge = KernelBridge()
    weights = {'A_hardware': 0.5, 'Φ_crystalline': 0.5}
    combined = bridge.combine_kernels(weights)

    x = {
        'A_hardware': {'latency': 10.0},
        'Φ_crystalline': {'wavefunction': np.array([1.0])}
    }
    y = {
        'A_hardware': {'latency': 10.0},
        'Φ_crystalline': {'wavefunction': np.array([1.0])}
    }

    val = combined(x, y)
    # 0.5 * exp(0) + 0.5 * 1.0 = 1.0
    assert val == 1.0
