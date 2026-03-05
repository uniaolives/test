import pytest
import numpy as np
from papercoder_kernel.cognition.acps_convergence import KatharosArkheMapping, HomeostasisRegime

def test_katharos_psi_projection():
    mapper = KatharosArkheMapping()
    psi = np.random.rand(10)
    vk = mapper.psi_to_vk(psi)

    assert vk.shape == (4,)
    # Verify Bio component maps to Psi[0]
    assert vk[0] == pytest.approx(psi[0])

def test_homeostasis_classification():
    regime = HomeostasisRegime()
    vk_ref = np.array([0.5, 0.5, 0.5, 0.5])

    # Safe regime
    vk_safe = np.array([0.55, 0.5, 0.45, 0.5])
    dk, label = regime.classify_acps(vk_safe, vk_ref)
    assert dk < 0.30
    assert "KATHARÓS" in label

    # Crisis regime
    vk_crisis = np.array([1.5, -0.5, 1.5, -0.5])
    dk, label = regime.classify_acps(vk_crisis, vk_ref)
    assert dk >= 0.70
    assert "CRISIS" in label

def test_delta_k_to_lambda_mapping():
    regime = HomeostasisRegime()
    # At perfect homeostasis, lambda should be PHI
    l_perfect = regime.map_delta_k_to_lambda(0.0)
    assert l_perfect == pytest.approx(regime.phi)

    # At threshold 0.3, lambda should still be high
    l_safe = regime.map_delta_k_to_lambda(0.3)
    assert 0.4 * regime.phi < l_safe < regime.phi

def test_inverse_mapping_psi_estimate():
    mapper = KatharosArkheMapping()
    vk = np.array([1.0, 0.8, 0.6, 0.4])
    psi_est = mapper.vk_to_psi_estimate(vk)

    assert psi_est.shape == (10,)
    # Re-projecting the estimate should return the original VK
    vk_rebuilt = mapper.psi_to_vk(psi_est)
    assert np.allclose(vk, vk_rebuilt)
