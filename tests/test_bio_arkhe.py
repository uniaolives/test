"""
Tests for Bio-Arkhe: Biological Emergence and Agent Behavior.
"""

import pytest
import numpy as np
from src.avalon.core.bio_arkhe import BioAgent, MorphogeneticField, ArkheGenome
from src.avalon.analysis.unified_particle_system import UnifiedParticleSystem

def test_bio_agent_bonding():
    # Two agents with high affinity (C > 0.5)
    genome = ArkheGenome(C=1.0, I=0.5, E=0.5, F=0.5)
    agent1 = BioAgent(0, np.array([0.0, 0.0, 0.0]), genome)
    agent2 = BioAgent(1, np.array([0.5, 0.0, 0.0]), genome)

    field = MorphogeneticField()
    all_agents = {0: agent1, 1: agent2}

    # Try bonding
    agent1.sense_and_act(field, all_agents)

    assert 1 in agent1.neighbors
    assert 0 in agent2.neighbors

def test_morphogenetic_field_gradient():
    field = MorphogeneticField(size=(10, 10, 10))
    # Injetar sinal manual
    field.signal_grid[8, 8, 8] = 10.0

    # Agente longe do sinal
    genome = ArkheGenome(C=0.5, I=0.5, E=1.0, F=0.5)
    agent = BioAgent(0, np.array([5.0, 5.0, 5.0]), genome)

    # Deve detectar gradiente (simplificado no mock atual)
    gradient = field.get_gradient(agent.position)
    assert not np.isnan(gradient).any()

def test_bio_genesis_transition():
    system = UnifiedParticleSystem(num_particles=20)
    system.set_mode("BIO_GENESIS")

    # Simula evolução
    for _ in range(50):
        system.update(0.1)

    data = system.get_particle_data()
    assert data['mode'] == "BIO_GENESIS"
    # Deve haver ligações se houver proximidade
    assert isinstance(data['bonds'], list)

def test_genome_personalities():
    # Agente rápido vs Agente lento
    g_fast = ArkheGenome(C=0.5, I=0.5, E=1.0, F=0.5)
    g_slow = ArkheGenome(C=0.5, I=0.5, E=0.1, F=0.5)

    a_fast = BioAgent(0, np.array([0.,0.,0.]), g_fast)
    a_slow = BioAgent(1, np.array([0.,0.,0.]), g_slow)

    # Force gradient
    class MockField:
        def get_gradient(self, pos): return np.array([1.0, 0.0, 0.0])

    a_fast.sense_and_act(MockField(), {})
    a_slow.sense_and_act(MockField(), {})

    assert a_fast.position[0] > a_slow.position[0]
