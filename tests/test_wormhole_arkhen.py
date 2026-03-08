import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.physics.wormhole_arkhen import WormholeArkhen, ConstitutionalField, WormholeMetric

def test_wormhole_initialization():
    """Test initialization with Planck length constraint"""
    # Should work
    wormhole = WormholeArkhen(throat_radius=1e-30)
    assert wormhole.r0 == 1e-30

    # Should fail (below Planck length)
    with pytest.raises(ValueError, match="Planck length"):
        WormholeArkhen(throat_radius=1e-36)

def test_geometry_generation():
    """Test Morris-Thorne metric generation and flaring condition"""
    wormhole = WormholeArkhen(throat_radius=1e-15)
    geom = wormhole.generate_geometry()

    assert isinstance(geom, WormholeMetric)
    assert geom.flaring == True
    assert len(geom.r) == 1000
    assert geom.r0 == 1e-15

    # Check asymptotic flatness (b/r -> 0)
    assert geom.b[-1] / geom.r[-1] < 0.1

def test_exotic_stress_energy():
    """Test computation of exotic energy density (rho < 0)"""
    wormhole = WormholeArkhen(throat_radius=1e-15)
    wormhole.generate_geometry()
    stress = wormhole.compute_exotic_stress_energy()

    assert stress['total_energy'] < 0
    assert stress['min_density'] < 0
    assert stress['exotic_condition'] == True

def test_stability_simulation():
    """Test stability under constitutional feedback"""
    wormhole = WormholeArkhen(throat_radius=1e-15)
    log = wormhole.stabilize(simulation_time=100)

    assert len(log) == 100
    final_pert = log[-1]['max_perturbation']
    # With feedback, perturbation should be controlled
    assert final_pert < 1.0

def test_traversal_metrics():
    """Test proper time and time dilation"""
    wormhole = WormholeArkhen(throat_radius=1e-15)
    wormhole.generate_geometry()
    metrics = wormhole.compute_traversal_time()

    assert metrics['proper_time'] > 0
    assert metrics['time_dilation'] > 0
    assert metrics['proper_length'] > 2 * 1e-15

def test_constitutional_violation():
    """Test strict mode violation"""
    field = ConstitutionalField(strict_mode=True)

    # P1 violation
    with pytest.raises(ValueError, match="P1: Naked singularity detected"):
        field.check_p1_no_singularities((np.inf, 1.0, 1.0, 1.0))

    # P2 violation
    with pytest.raises(ValueError, match="P2: Excessive redshift"):
        field.check_p2_causality(np.array([11.0]), np.array([1.0]), np.array([1.0]))

def test_full_deployment():
    """Test the complete deployment pipeline"""
    wormhole = WormholeArkhen(throat_radius=1e-15)
    result = wormhole.deploy()

    assert result['navigable'] == True
    assert 'geometry' in result
    assert 'stress_energy' in result
    assert 'stability' in result
    assert 'traversal' in result
