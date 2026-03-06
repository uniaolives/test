import pytest
from gateway.app.physics.simulators import QuantumSimulator
from gateway.app.physics.triggers import ArkheTrigger

def test_quantum_simulator():
    sim = QuantumSimulator(tau_c=1e-6)
    # Test adaptive Hilbert space
    n_dim = sim.adaptive_hilbert_space(xi=2.0)
    assert n_dim >= 30

    # Test orbital decoherence
    tau = sim.orbital_decoherence(h_orbit=400e3)
    assert tau < 1e-6 # Must be restricted by ionosphere

def test_lhc_trigger():
    # Use time scale of picoseconds as in LHC
    trigger = ArkheTrigger(time_resolution=1e-13)
    jets = [
        {'pt': 100, 'time': 1e-12, 'eta': 0.1, 'phi': 0.1},
        {'pt': 100, 'time': 2e-12, 'eta': 0.15, 'phi': 0.15}
    ]
    # i=0, j=1 -> dt = -1e-12. -1e-12 < -1e-13 is True.
    # Score should be -log10(1e-12) * ... = 12 * ...
    result = trigger.evaluate_event(jets)
    assert result['n_violations'] == 1
    assert result['arkhe_score'] > 0
