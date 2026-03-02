# tests/test_glp_linear_a.py
import pytest
import torch
import numpy as np
from papercoder_kernel.glp.model import BCD_GLPLinearA, HarmonicConfinement
from papercoder_kernel.glp.training import QuantumActionLoss, analyze_confinement
from papercoder_kernel.glp.integration import LinearAToPaperCoder

def test_harmonic_confinement():
    well = HarmonicConfinement(max_n=4, sigma=1.0, resolution=100)
    positions = torch.linspace(-1, 1, 10).unsqueeze(0) # [1, 10]
    amplitudes = torch.tensor([[1.0, 0.0, 0.0, 0.0]]) # Only ground state |0>

    wf = well(positions, amplitudes)
    assert wf.shape == (1, 10)
    # Ground state of harmonic oscillator is a Gaussian, should be max at center
    assert torch.argmax(wf) == 4 or torch.argmax(wf) == 5

def test_bcd_model_forward():
    vocab_size = 50
    model = BCD_GLPLinearA(vocab_size=vocab_size, hidden_dim=64)
    sign_ids = torch.randint(1, vocab_size, (2, 20)) # [batch=2, seq=20]

    outputs = model(sign_ids, return_wavefunction=True)

    assert 'sign_logits' in outputs
    assert outputs['sign_logits'].shape == (2, 20, vocab_size)
    assert 'well_states' in outputs
    # [batch, n_wells, seq, hidden]
    assert outputs['well_states'].shape == (2, 6, 20, 64)
    assert 'tunneled_states' in outputs
    assert outputs['tunneled_states'].shape == (2, 6, 20, 64)

def test_quantum_action_loss():
    loss_fn = QuantumActionLoss()
    vocab_size = 10
    logits = torch.randn(2, 5, vocab_size)
    targets = torch.randint(1, vocab_size, (2, 5))

    outputs = {
        'sign_logits': logits,
        'tunneling_strength': torch.tensor(0.5)
    }
    model_states = {
        'tunneled_states': torch.randn(2, 3, 5, 16)
    }

    loss, loss_dict = loss_fn(outputs, targets, model_states)
    assert loss > 0
    assert 'potential' in loss_dict
    assert 'kinetic' in loss_dict
    assert 'tunnel' in loss_dict

def test_analyze_confinement():
    # Matrix with equally spaced eigenvalues (harmonic)
    eigenvals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Create a symmetric matrix with these eigenvalues
    Q, _ = np.linalg.qr(np.random.randn(5, 5))
    cooc = Q @ np.diag(eigenvals) @ Q.T

    analysis = analyze_confinement(cooc)
    assert analysis['confinement_regime'] == 'harmonic'
    assert pytest.approx(analysis['mean_spacing_ratio'], rel=1e-2) == 1.0

def test_integration_manifold():
    vocab_size = 20
    model = BCD_GLPLinearA(vocab_size=vocab_size, hidden_dim=32)
    integration = LinearAToPaperCoder(model)

    sign_ids = torch.randint(1, vocab_size, (5, 10))
    manifold = integration.extract_manifold(sign_ids)

    assert manifold['tablet_repr'].shape == (5, 32)
    assert manifold['scale_probabilities'].shape == (5, 6, 32)

def test_procrustes_alignment():
    integration = LinearAToPaperCoder(None)
    A = np.random.randn(10, 8)
    # B is A rotated
    R_true, _ = np.linalg.qr(np.random.randn(8, 8))
    B = A @ R_true

    R_est = integration.procrustes_alignment(A, B)
    assert R_est.shape == (8, 8)

    # Check if A @ R_est matches B
    B_est = (A - np.mean(A, axis=0)) @ R_est
    B_target = B - np.mean(B, axis=0)
    assert np.allclose(B_est, B_target, atol=1e-5)
