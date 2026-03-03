# tests/test_quantum_network.py
import pytest
from arkhe.quantum_network import QuantumNetwork, QuantumNode, get_initial_network

def test_quantum_node_activation():
    net = get_initial_network()
    # Add latent node
    net.add_node(QuantumNode("QN-04", "PREV_001", 0.0, 0.8, 0.6, False))

    assert net.nodes["QN-04"].is_active == False

    net.activate_node("QN-04", 0.04)
    assert net.nodes["QN-04"].is_active == True
    assert net.nodes["QN-04"].omega == 0.04

def test_network_range():
    net = get_initial_network()
    # Initial range: min 0.0 (WP1), max 0.07 (DVM-1) -> 0.07
    assert net.calculate_max_range() == 0.07

    # Extend range
    net.add_node(QuantumNode("QN-05", "EXT", 0.11, 0.9, 0.5, True))
    assert net.calculate_max_range() == 0.11

def test_key_integrity():
    net = get_initial_network()
    assert net.verify_key_integrity() == True

    # Simulate interception
    net.nodes["QN-01"].epsilon_key = 1.0
    assert net.verify_key_integrity() == False

def test_bell_violation():
    net = get_initial_network()
    # Initial Bell test (CHSH = 2.414)
    assert net.run_bell_test() == 2.414

    # Activate Kernel node
    net.activate_kernel_node()
    # Bell test should increase (CHSH = 2.428)
    assert net.run_bell_test() == 2.428

def test_kernel_integration():
    net = get_initial_network()
    kernel = net.activate_kernel_node()
    assert kernel.id == "QN-06"
    assert kernel.is_active == True
    assert kernel.omega == 0.12
    assert net.nodes["QN-06"].phi == 0.94
