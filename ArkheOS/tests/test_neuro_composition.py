
import pytest
from arkhe.neuro_composition import NeuroCompositionEngine

def test_neuro_composition_engagement():
    engine = NeuroCompositionEngine()

    # Target DVM-1 subspace
    result = engine.process_stimulus(0.07, hesitation_phi=0.10)
    assert result == "déjà vu"
    assert engine.belief_manager.current_belief == 0.07
    assert engine.subspaces[0.07].engagement_count == 1

def test_belief_update_closest_match():
    engine = NeuroCompositionEngine()

    # Target something slightly off, should find closest
    result = engine.process_stimulus(0.124, hesitation_phi=0.05)
    assert result == "consciência"
    assert engine.belief_manager.current_belief == 0.12

def test_belief_history():
    engine = NeuroCompositionEngine()
    engine.process_stimulus(0.03, 0.1)
    engine.process_stimulus(0.21, 0.2)

    history = engine.belief_manager.belief_history
    assert len(history) == 2
    assert history[0][0] == 0.03
    assert history[1][0] == 0.21
