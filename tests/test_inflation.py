# tests/test_inflation.py
import pytest
import numpy as np
from papercoder_kernel.dynamics.inflation import ScaleAwareInflation

def test_inflation_init():
    n_scales = 15
    sai = ScaleAwareInflation(n_scales)
    assert sai.n == n_scales
    assert sai.rho0 == 1.02
    assert sai.gamma == 0.5

def test_inflation_factor_zero_variance():
    n_scales = 5
    sai = ScaleAwareInflation(n_scales)
    # Manual set prior_var to zero
    sai.prior_var = np.zeros(n_scales)
    factor = sai.inflation_factor(0)
    assert factor == 1.02

def test_apply_inflation_increasing_variance():
    n_scales = 3
    sai = ScaleAwareInflation(n_scales, base_inflation=1.0)
    # Ensemble with increasing variance per scale
    # Scale 0: constant
    # Scale 1: small var
    # Scale 2: large var
    ensemble = np.array([
        [1.0, 0.9, 0.0],
        [1.0, 1.1, 10.0],
        [1.0, 1.0, 5.0]
    ])

    inflated = sai.apply_inflation(ensemble)

    # Scale 0 should have rho = rho0 * (1 + gamma * 0) = 1.0
    assert np.allclose(inflated[:, 0], ensemble[:, 0])

    # Check that scale 2 inflated more than scale 1 relative to mean
    # Variances
    vars = np.var(ensemble, axis=0, ddof=1)
    mean_var = np.mean(vars)

    rho1 = 1.0 * (1 + 0.5 * vars[1] / mean_var)
    rho2 = 1.0 * (1 + 0.5 * vars[2] / mean_var)

    assert rho2 > rho1

    mean = np.mean(ensemble, axis=0)
    expected_member0_scale1 = mean[1] + rho1 * (ensemble[0, 1] - mean[1])
    assert np.isclose(inflated[0, 1], expected_member0_scale1)
