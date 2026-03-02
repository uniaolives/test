"""
Tests for Cosmic Synthesis: Unified Intelligence System and Constraint-Based AI.
"""

import pytest
import numpy as np
import torch
from src.avalon.core.unified_intelligence_theory import UnifiedIntelligenceSystem
from src.avalon.analysis.cognitive_light_cone import CognitiveLightCone
from src.avalon.analysis.neuro_metasurface import NeuroMetasurfaceController
from src.avalon.core.schmidt_bridge import SchmidtBridgeHexagonal
from src.avalon.core.celestial_helix import CosmicDNAHelix
from src.avalon.analysis.arkhe_theory import ArsTheurgiaGoetia
from src.avalon.core.unified_ai import UnifiedAI, ConstraintDiscoveryNetwork

def test_unified_intelligence_system():
    cone = CognitiveLightCone()
    meta = NeuroMetasurfaceController()
    arkhe = SchmidtBridgeHexagonal(np.array([0.8, 0.1, 0.05, 0.02, 0.02, 0.01]))
    celestial = CosmicDNAHelix()
    goetia = ArsTheurgiaGoetia()

    system = UnifiedIntelligenceSystem(
        cognitive_cone=cone,
        neuro_em_controller=meta,
        arkhe_state=arkhe,
        celestial_modulator=celestial,
        goetic_navigator=goetia
    )

    metrics = system.unified_intelligence_metric()
    assert 'unified_intelligence' in metrics
    assert 0.0 <= metrics['unified_intelligence'] <= 1.0
    assert 'interpretation' in metrics

def test_unified_ai_forward():
    model = UnifiedAI(state_dim=10, constraint_dim=5, arkhe_rank=6)
    state = torch.randn(1, 10)
    output = model(state, attention=0.9)

    assert 'action' in output
    assert output['action'].shape == (1, 10)
    assert 'coherence' in output
    assert 0.0 <= output['coherence'].item() <= 1.0
    assert 'intelligence_estimate' in output

def test_constraint_discovery():
    network = ConstraintDiscoveryNetwork(input_dim=10, constraint_dim=3)
    state = torch.randn(2, 10)
    constraints = network(state)
    assert constraints.shape == (2, 3)
    assert torch.all(constraints >= 0) and torch.all(constraints <= 1)

def test_cognitive_light_cone():
    cone = CognitiveLightCone()
    metric = cone.calculate_intelligence_metric()
    assert metric['intelligence_score'] > 0

    grad = cone._calculate_constraint_gradient(np.array([1.0, 0.0, 0.0, 0.0]))
    assert grad.shape == (4,)
