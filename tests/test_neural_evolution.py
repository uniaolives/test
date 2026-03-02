"""
Tests for Neural Evolution and Constraint Discovery in Bio-Arkhe.
"""

import pytest
import numpy as np
from src.avalon.core.bio_arkhe import BioAgent, ArkheGenome, MorphogeneticField
from src.avalon.core.constraint_engine import ConstraintLearner

def test_constraint_learner_initialization():
    learner = ConstraintLearner(agent_id=0)
    assert len(learner.weights) == 4
    assert learner.successful_bonds == 0
    assert learner.toxic_bonds == 0

def test_evaluate_and_learn():
    learner = ConstraintLearner(agent_id=0, learning_rate=1.0)
    partner_genome = ArkheGenome(C=1.0, I=0.0, E=0.0, F=0.0)

    # Initial evaluation
    initial_score, _ = learner.evaluate_partner(partner_genome)

    # Positive feedback interaction
    learner.learn_from_interaction(partner_genome, partner_id=1, energy_delta=0.1, timestamp=0)

    # Score should increase for similar partners
    new_score, _ = learner.evaluate_partner(partner_genome)
    assert new_score > initial_score
    assert learner.successful_bonds == 1

def test_brain_state_labels():
    learner = ConstraintLearner(agent_id=0)
    # Force seeking chemistry
    learner.weights = np.array([1.0, 0.0, 0.0, 0.0])
    assert "+C" in learner.get_weights_description()

    # Force avoiding energy
    learner.weights = np.array([0.0, 0.0, -1.0, 0.0])
    assert "-E" in learner.get_weights_description()

def test_bio_agent_brain_integration():
    genome = ArkheGenome(C=0.5, I=1.0, E=0.5, F=0.5)
    agent = BioAgent(0, np.array([0.,0.,0.]), genome)

    assert isinstance(agent.brain, ConstraintLearner)
    assert agent.brain.base_learning_rate > 0.1 # I=1.0 -> lr = 0.05 + 0.2 = 0.25

def test_metabolic_feedback_loop():
    from src.avalon.analysis.unified_particle_system import UnifiedParticleSystem
    system = UnifiedParticleSystem(num_particles=2)
    system.set_mode("BIO_GENESIS")
    system.transition_progress = 1.0 # Force transition completion

    # Force a connection
    a1, a2 = system.agents[0], system.agents[1]
    a1.neighbors.append(a2.id)
    a2.neighbors.append(a1.id)

    # Run update - should trigger feedback
    a1.health = 1.1 # artificially increase
    system.update(0.1)

    # The brain should have learned something
    assert a1.brain.successful_bonds + a1.brain.toxic_bonds > 0
