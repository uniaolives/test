"""
Tests for Unified Particle System and Visual Quantification.
"""

import pytest
import numpy as np
from src.avalon.analysis.unified_particle_system import UnifiedParticleSystem, get_hypercore_pos
from src.avalon.analysis.visual_quantification import VisualQuantificationEngine
from src.avalon.core.arkhe import NormalizedArkhe

def test_particle_system_initialization():
    system = UnifiedParticleSystem(num_particles=50)
    assert len(system.particles) == 50
    assert system.current_mode == "MANDALA"

def test_mode_transition():
    system = UnifiedParticleSystem(num_particles=10)
    system.set_mode("DNA")
    assert system.target_mode == "DNA"
    assert system.current_mode == "MANDALA"

    # Run some updates
    for _ in range(10):
        system.update(0.1)

    assert system.transition_progress > 0

def test_hypercore_projection():
    # Test that it returns a 3D vector
    pos = get_hypercore_pos(0, 120, 0.0)
    assert pos.shape == (3,)
    assert not np.isnan(pos).any()

def test_visual_quantification():
    system = UnifiedParticleSystem(num_particles=120)
    engine = VisualQuantificationEngine(system)

    # C dominant Arkhe
    arkhe_c = NormalizedArkhe(C=0.8, I=0.1, E=0.1, F=0.0)
    params = engine.quantify_arkhe_state(arkhe_c)
    assert params['mode'] == "MANDALA"

    # Apply to visualization
    engine.apply_to_visualization(arkhe_c)
    assert system.target_mode == "MANDALA"

    # I dominant Arkhe
    arkhe_i = NormalizedArkhe(C=0.1, I=0.8, E=0.1, F=0.0)
    engine.apply_to_visualization(arkhe_i)
    assert system.target_mode == "DNA"
