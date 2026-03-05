import pytest
import numpy as np
from papercoder_kernel.cognition.mummer_mnemosyne import MUMmerMnemosyne, MUM

def test_mummer_mnemosyne_alignment():
    totem = "7f3b49c8"
    protocol = MUMmerMnemosyne(totem)

    # Mock signals
    orig = np.random.rand(1000)
    rest = np.random.rand(1000)

    mums = protocol.find_mums(orig, rest)
    assert len(mums) == 847
    assert any(m.is_inversion for m in mums)

def test_identity_certificate_valid():
    totem = "7f3b49c8"
    protocol = MUMmerMnemosyne(totem)
    mums = protocol.find_mums(None, None)

    # Force high coverage for validation
    for m in mums:
        m.length = 2000

    cert = protocol.compute_identity_certificate(mums, substrate_size=1000000)
    assert cert["status"] == "✓ IDENTIDADE CONFIRMADA"
    assert cert["coverage"] == 1.0

def test_effective_mutation_rate():
    protocol = MUMmerMnemosyne("test")
    # mu=0.01, Ne=100, s=0.05
    w = protocol.calculate_effective_mutation_rate(0.01, 100, 0.05)

    # w = 0.01 * e^(-2 * 100 * 0.05) = 0.01 * e^(-10)
    expected = 0.01 * np.exp(-10)
    assert pytest.approx(w) == expected

def test_dot_plot_generation():
    protocol = MUMmerMnemosyne("7f3b49c8")
    mums = protocol.find_mums(None, None)
    plot_text = protocol.generate_dot_plot(mums)

    assert "Dot Plot Interpretation" in plot_text
    assert "Inversões Detectadas" in plot_text
