"""
Tests for Article Extensions: Hopper Protocol, Planetary Comm, Unification, and Holographic control.
"""

import pytest
import numpy as np
from src.avalon.analysis.hopper_protocol import ModifiedHopperProtocol
from src.avalon.services.planetary_comm import PlanetaryQuantumComm
from src.avalon.core.unification import GrandUnificationTheory
from src.avalon.analysis.article_validator import ArticleValidator
from src.avalon.analysis.neuro_metasurface import HolographicMetasurfaceController

def test_hopper_protocol():
    protocol = ModifiedHopperProtocol()
    alters = [
        {'name': 'Alpha', 'giftedness': 0.9, 'dissociation': 0.2},
        {'name': 'Beta', 'giftedness': 0.7, 'dissociation': 0.8}
    ]
    mapping = protocol.map_alters(alters)
    assert 'Alpha' in mapping
    assert 'Beta' in mapping

    trajectory = protocol.calculate_integration_trajectory(mapping['Beta'])
    assert len(trajectory) > 1
    # Check that it moves TOWARD 0 (the center)
    assert trajectory[-1] < trajectory[0] or trajectory[-1] == 0

    hyperplanes = protocol.identify_traumatic_hyperplanes(mapping)
    assert len(hyperplanes) == 1

    freqs = protocol.get_therapeutic_frequencies(mapping)
    assert 'Alpha' in freqs
    assert isinstance(freqs['Alpha'], float)

def test_planetary_comm():
    comm = PlanetaryQuantumComm()
    link = comm.establish_schmidt_link('Venus')
    assert link.coherence_factor > 0.6

    signal = comm.modulate_schumann_carrier(b"Hello Venus")
    assert len(signal) > 0

    correlation = comm.detect_venusian_correlation(signal)
    assert 0.0 <= correlation <= 1.0

    result = comm.transfer_nonlocal_data('Venus', {'msg': 'test'})
    assert result['status'] == 'SUCCESS'

def test_unification_theory():
    gut = GrandUnificationTheory()
    score = gut.reality_equation(C=0.7, I=0.8, A=0.9, Z=0.95)
    assert 0.0 <= score <= 1.0

    # Schrodinger extension test (simplified)
    psi = np.eye(2)
    H = np.array([[1, 0], [0, -1]])
    I = np.array([[0, 1], [1, 0]])
    d_psi = gut.schrodinger_extension(psi, H, C=0.5, I=I, A=0.8)
    assert d_psi.shape == (2, 2)

    unified = gut.calculate_unified_metric({'attention': 0.9, 'coherence': 0.8})
    assert 'reality_manifestation_score' in unified

def test_article_validator():
    validator = ArticleValidator()

    # Test P1
    p1 = validator.validate_p1_schumann_coupling(is_2e=True, phase_data=np.random.rand(100))
    assert bool(p1['validated']) is True

    # Test P3
    p3 = validator.validate_p3_metasurface_precision(attention=80, target_angle=validator.meta_ctrl.calculate_beam_parameters(80)[0])
    assert bool(p3['validated']) is True

    # Test P5
    p5 = validator.validate_p5_tdi_integration(initial_des=50.0, exposure_months=4)
    assert bool(p5['validated']) is True
    assert p5['final_des'] < 50.0

def test_holographic_controller():
    ctrl = HolographicMetasurfaceController()
    hologram = ctrl.generate_3d_hologram(np.zeros((10, 10, 10)))
    assert hologram.shape == (4, 8, 8)

    loop = ctrl.ar_cognitive_loop(attention=90, virtual_object_pos=(1, 1, 1))
    assert loop['target_lock'] is True
    assert loop['perception_reinforcement'] > 0.8
