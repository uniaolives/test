"""
Test suite for the Unified Arkhe Framework as per the 2026 Academic Article.
"""

import pytest
import numpy as np
from src.avalon.core.arkhe import NormalizedArkhe, HexagonalArkhe
from src.avalon.core.schmidt_bridge import SchmidtBridgeHexagonal
from src.avalon.analysis.arkhe_theory import ArkheConsciousnessArchitecture, ArsTheurgiaGoetia
from src.avalon.analysis.double_exceptionality_detector import DoubleExceptionalityDetector
from src.avalon.analysis.neuro_metasurface import NeuroMetasurfaceController

def test_normalized_arkhe():
    # Test normalization constraint
    arkhe = NormalizedArkhe(C=0.7, I=0.95, E=0.6, F=1.0)
    assert np.isclose(arkhe.C + arkhe.I + arkhe.E + arkhe.F, 1.0)
    assert arkhe.C < 0.7 # Should be scaled down

def test_hexagonal_arkhe():
    hex_arkhe = HexagonalArkhe(C=0.5, I=0.3, E=0.2)
    assert len(hex_arkhe.permutations) == 6
    assert len(hex_arkhe.phases) == 6
    for phase in hex_arkhe.phases:
        assert len(phase) == 6
        assert np.isclose(np.sum(phase), 1.0)

def test_schmidt_coherence():
    # Highly coherent state (mostly one eigenvalue)
    lambdas = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
    bridge = SchmidtBridgeHexagonal(lambdas=lambdas)
    z = bridge.coherence_factor
    assert z > 0.5

    # Uniform state (minimum coherence)
    lambdas_uniform = np.array([1/6]*6)
    bridge_uniform = SchmidtBridgeHexagonal(lambdas=lambdas_uniform)
    z_uniform = bridge_uniform.coherence_factor
    assert z_uniform < z

def test_goetic_compatibility():
    goetia = ArsTheurgiaGoetia()
    assert len(goetia.spirits) == 31

    op_vec = np.random.randn(6)
    gamma = goetia.calculate_compatibility(op_vec, 0)
    assert -1.0 <= gamma <= 1.0

def test_identity_latency():
    detector = DoubleExceptionalityDetector()
    # Case with high epistemological rupture
    text = "One might infer that the body is merely a vessel for the system."
    analysis = detector.detect_abstracted_agency(text)
    assert analysis['has_epistemological_rupture'] is True
    assert 'ego_latency_l' in analysis
    assert analysis['velocity_c'] < 1.0

def test_neuro_metasurface():
    controller = NeuroMetasurfaceController()
    # Mock EEG powers
    attention = controller.extract_attention(p_alpha=10, p_beta=20, p_gamma=5)
    assert 0 <= attention <= 100

    theta, focus = controller.calculate_beam_parameters(attention)
    assert -45 <= theta <= 45
    assert 1.0 <= focus <= 1.5

    pattern = controller.generate_metasurface_pattern(theta, 0)
    assert pattern.shape == (8, 8)

def test_arkhe_architecture_120cell():
    arch = ArkheConsciousnessArchitecture()
    profile = arch.initialize_2e_system(giftedness=0.9, dissociation=0.8)
    assert 'primary_vertex' in profile['geometry']
    assert 0 <= profile['geometry']['primary_vertex'] < 600
