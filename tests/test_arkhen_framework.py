import pytest
import numpy as np
from papercoder_kernel.cognition.primary_evaluation import EpigeneticModulation, QualicDynamics, MonteCarloValidator

def test_epigenetic_modulation():
    mod = EpigeneticModulation(epsilon=0.5)
    # v_eps = 1 + 3 * 0.5 = 2.5
    assert mod.vulnerability_factor() == 2.5
    # eps_acc = 1 - 0.3 * 0.5 = 0.85
    assert mod.cumulative_attenuation() == 0.85

def test_qualic_dynamics_ode():
    dyn = QualicDynamics(alpha_q=0.1, gamma_q=0.05)

    # Positive formation: low delta_k (0.1 < 0.3)
    # formation = 0.1 * 1.0 * (1 - 0.1) * (1 - 0.5 * 0.0) * (1 - 0.5) = 0.1 * 0.9 * 0.5 = 0.045
    # destruction = 0.0
    val_inc = dyn.model(q=0.5, t=0, p_eff=1.0, delta_k=0.1, c_neuro=0.0, dd=0.0, v_eps=1.0)
    assert val_inc > 0
    assert pytest.approx(val_inc) == 0.045

    # Positive destruction: high delta_k (0.8 > 0.3)
    # formation = 0.1 * 1.0 * (1 - 0.8) * (1 - 0) * (1 - 0.5) = 0.1 * 0.2 * 0.5 = 0.01
    # destruction = 0.05 * 0.5 * (0.8 - 0.3) * (1 + 0) * 1 = 0.05 * 0.5 * 0.5 = 0.0125
    # total = 0.01 - 0.0125 = -0.0025
    val_dec = dyn.model(q=0.5, t=0, p_eff=1.0, delta_k=0.8, c_neuro=0.0, dd=0.0, v_eps=1.0)
    assert val_dec < 0
    assert pytest.approx(val_dec) == -0.0025

def test_monte_carlo_mock():
    validator = MonteCarloValidator(n_simulations=100)
    report = validator.run_validation()
    assert report["status"] == "VALIDATED"
    assert report["predictions_matched"] == 7
