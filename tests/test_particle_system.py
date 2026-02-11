"""
Tests for Unified Particle System and Bio-Genesis Engine v3.0.
"""

import pytest
import numpy as np
from src.avalon.analysis.unified_particle_system import UnifiedParticleSystem, get_hypercore_pos
from src.avalon.analysis.visual_quantification import VisualQuantificationEngine
from src.avalon.core.arkhe import NormalizedArkhe
from src.avalon.core.particle_system import BioGenesisEngine

def test_particle_system_initialization():
    system = UnifiedParticleSystem(num_particles=50)
    assert len(system.particles) == 50
    assert system.current_mode == "MANDALA"

def test_bio_genesis_engine_initialization():
    engine = BioGenesisEngine(num_agents=100)
    assert len(engine.agents) == 100
    assert len(engine.signal_sources) > 0

def test_spatial_hash_query():
    from src.avalon.core.particle_system import SpatialHash
    # Larger cell size to separate agents
    sh = SpatialHash(cell_size=20.0)
    sh.insert(1, np.array([5.0, 5.0, 5.0]))    # cell (0,0,0)
    sh.insert(2, np.array([45.0, 45.0, 45.0])) # cell (2,2,2)
    sh.insert(3, np.array([95.0, 95.0, 95.0])) # cell (4,4,4)

    # Near agent 1
    near_1 = sh.query(np.array([6.0, 6.0, 6.0]), radius=5.0)
    assert 1 in near_1
    assert 2 not in near_1

    # Far from everything
    none = sh.query(np.array([200.0, 200.0, 200.0]), radius=5.0)
    assert len(none) == 0

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
