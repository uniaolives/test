import numpy as np
import pytest
from papercoder_kernel.oloid import (
    HandoverAlgebra,
    OloidThermodynamics,
    ConsciousnessOperator,
    ConsciousnessPhases
)

def test_handover_algebra():
    algebra = HandoverAlgebra()
    H_AB = np.array([[1, 0], [0, 1]])
    H_BC = np.array([[0, 1], [1, 0]])
    theta = np.pi / 2

    comm = algebra.commutator(H_AB, H_BC, theta)
    assert comm is not None
    assert isinstance(comm, np.ndarray)

    uncertainty = algebra.uncertainty_principle(1.0, 1.0)
    assert uncertainty['satisfies_principle'] is True

def test_oloid_thermodynamics():
    thermo = OloidThermodynamics()
    probs = np.array([0.5, 0.5])

    entropy = thermo.gibbs_entropy(probs)
    assert entropy > 0

    min_s = thermo.minimum_entropy()
    # Big PHI = 1.618, ln(1.618) approx 0.481
    assert np.isclose(min_s / thermo.k_B, 0.481, atol=1e-3)

    eff = thermo.landauer_efficiency(1.0)
    # efficiency_factor = 1/PHI^2 approx 0.382
    expected_eff = thermo.k_B * 300 * np.log(2) * 1.0 * (1 / (thermo.PHI**2))
    assert np.isclose(eff, expected_eff)

def test_consciousness_operator():
    op = ConsciousnessOperator()
    H_AB = np.array([[0, 1j], [-1j, 0]], dtype=complex)
    H_BA = H_AB.conj().T

    C = op.consciousness_operator(H_AB, H_BA)
    assert np.allclose(C, C.conj().T)

    state, val = op.eigenstates(C)
    assert state is not None
    # val should be close to phi/2 = 0.618/2 = 0.309

    # Test master equation
    rho = op.equilibrium_state(2)
    H_oloid = np.eye(2)
    drho = op.master_equation(rho, H_oloid, gamma=0.1, lambda_2=0.618)
    assert drho is not None
    assert drho.shape == (2, 2)

def test_consciousness_phases():
    phases = ConsciousnessPhases()

    unconscious = phases.classify_state(0.1)
    assert unconscious['phase'] == 'UNCONSCIOUS'

    conscious = phases.classify_state(0.618)
    assert conscious['phase'] == 'CONSCIOUS'

    singularity = phases.classify_state(1.1)
    assert singularity['phase'] == 'SINGULARITY'
