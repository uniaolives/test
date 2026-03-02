# tests/test_merkabah_firewall.py
import pytest
from papercoder_kernel.merkabah.topological.firewall import ChiralQuantumFirewall

def test_firewall_access_granted():
    firewall = ChiralQuantumFirewall(target_node="TestNode", gap_meV=0.5, winding_number=2)
    packet = {
        'phase': 2,
        'energy_meV': 0.5
    }
    allowed, msg = firewall.validate_handover(packet)
    assert allowed == True
    assert "ACCESS_GRANTED" in msg

def test_firewall_invalid_phase():
    firewall = ChiralQuantumFirewall(target_node="TestNode", winding_number=2)
    packet = {'phase': 3}
    allowed, msg = firewall.validate_handover(packet)
    assert allowed == False
    assert "Invalid Winding Number" in msg

def test_firewall_energy_outside_gap():
    firewall = ChiralQuantumFirewall(target_node="TestNode", gap_meV=0.5)
    packet = {
        'phase': 2,
        'energy_meV': 0.8
    }
    allowed, msg = firewall.validate_handover(packet)
    assert allowed == False
    assert "outside chiral gap" in msg

def test_firewall_tolerance():
    firewall = ChiralQuantumFirewall(target_node="TestNode", gap_meV=0.5)
    # 1% tolerance: 0.5 * 0.01 = 0.005. 0.504 should be allowed.
    packet = {
        'phase': 2,
        'energy_meV': 0.504
    }
    allowed, msg = firewall.validate_handover(packet)
    assert allowed == True
