# tests/test_hydrodynamic_propulsion.py

import pytest
import numpy as np
from UrbanSkyOS.intelligence.hydrodynamic_propulsion import QuantumHydrodynamicEngine

def test_quantum_potential_calculation():
    engine = QuantumHydrodynamicEngine(mass=1.0, hbar=1.0)
    x = np.linspace(-5, 5, 100)
    dx = x[1] - x[0]

    # Pacote gaussiano: rho = exp(-x^2/2) / sqrt(2*pi)
    rho = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    Q = engine.compute_quantum_potential(rho, dx)

    # Para gaussiana unitária (sigma=1), Q = 1/4 * (1 - x^2/2)
    # se hbar=1, m=1.

    # Verificar no centro (x=0)
    assert Q[50] == pytest.approx(0.25, rel=0.1)

def test_propulsion_modulation():
    engine = QuantumHydrodynamicEngine(mass=1e-6)
    result = engine.modulate_for_propulsion(
        base_sigma=1e-6,
        modulation_freq=1e4,
        modulation_amp=0.1,
        duration=0.01,
        dt=0.0001
    )

    assert 'total_momentum' in result
    assert result['max_force'] > 0
    assert len(result['forces']) > 0

def test_coherence_calculation():
    engine = QuantumHydrodynamicEngine()
    x = np.linspace(-5, 5, 100)
    dx = x[1] - x[0]
    rho = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    C = engine.compute_coherence(rho, dx)
    assert 0 <= C <= 1
