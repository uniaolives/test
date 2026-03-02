
import pytest
from arkhe.topology import TopologyEngine, TopologicalQubit

def test_chern_numbers():
    assert TopologyEngine.calculate_chern_number(0.00) == 0.0
    assert TopologyEngine.calculate_chern_number(0.05) == 1.0
    assert TopologyEngine.calculate_chern_number(0.07) == 0.333

def test_quantum_metric():
    # g = 1 - overlap^2
    # For overlap=0.94, g = 1 - 0.8836 = 0.1164
    g = TopologyEngine.calculate_quantum_metric(0.94)
    assert abs(g - 0.1164) < 0.001

def test_phase_report():
    report = TopologyEngine.get_phase_report(0.07)
    assert "Fracionário" in report.label
    assert report.chern_number == 0.333

def test_topological_qubit():
    qubit = TopologicalQubit()
    assert qubit.pulse_gate(0.02) is True
