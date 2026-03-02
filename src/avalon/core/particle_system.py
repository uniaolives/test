"""
BIO-GÊNESE: Active Component Assembly Engine.
Substitui o sistema de partículas estáticas por agentes autônomos.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
from .bio_arkhe import BioAgent, ArkheGenome, MorphogeneticField

# Constantes do Sistema Vivo
INITIAL_POPULATION = 800
FIELD_SIZE = (100, 100, 100)
SPAWN_RADIUS = 40
MUTATION_RATE = 0.01

@dataclass
class BioState:
    """Estado global do ecossistema"""
    time_step: int = 0
    total_energy: float = 0.0
    structure_coherence: float = 0.0
    signal_diversity: float = 0.0

class BioParticleEngine:
    """
    Motor principal que orquestra o ecossistema de agentes.
    Implementa os 5 princípios da inteligência biológica.
    """

    def __init__(self, num_agents: int = INITIAL_POPULATION):
        self.field = MorphogeneticField(size=FIELD_SIZE)
        self.agents: Dict[int, BioAgent] = {}
        self.agent_counter = 0
        self.state = BioState()
        self.signals: Dict[Tuple[int, int, int], float] = {}

        self._create_primordial_soup(num_agents)
        self._add_signal_source(np.array(FIELD_SIZE) // 2, 15.0)

    def _create_primordial_soup(self, num_agents: int):
        center = np.array(FIELD_SIZE) // 2
        for i in range(num_agents):
            theta = random.random() * 2 * np.pi
            phi = random.random() * np.pi
            r = random.random() * SPAWN_RADIUS
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)

            genome = ArkheGenome(
                C=random.uniform(0.3, 0.9),
                I=random.uniform(0.1, 0.7),
                E=random.uniform(0.4, 1.0),
                F=random.uniform(0.1, 0.5)
            )

            agent = BioAgent(
                id=self.agent_counter,
                position=np.array([x, y, z], dtype=np.float32),
                genome=genome
            )
            self.agents[self.agent_counter] = agent
            self.agent_counter += 1

    def _add_signal_source(self, position: np.ndarray, strength: float):
        x, y, z = position.astype(int)
        self.signals[(x, y, z)] = strength

    def step(self, dt: float = 0.016):
        """Avança um tick da vida"""
        self.state.time_step += 1

        # 0. Guarda saúde anterior
        for agent in self.agents.values():
            agent.prev_health = agent.health

        # 1. Atualiza campo morfogenético
        self.field.update_field(list(self.agents.values()))

        # 2. Atualiza cada agente
        for agent in list(self.agents.values()):
            if agent.health <= 0: continue
            sensory = agent.sense_environment(self.field)
            action = agent.decide_action(sensory, self.agents)
            agent.update_state(action, dt)

            # Recupera energia em áreas de alto sinal
            if sensory['signal'] > 5.0:
                agent.health = min(1.0, agent.health + 0.005)
            # Consumo base
            agent.health -= 0.001 * (1.0 - agent.genome.E)

        # 3. Processa interações inteligentes
        self._process_smart_interactions()

        # 4. Fase de Aprendizado
        self._metabolic_feedback()

        # 5. Limpeza e métricas
        self._purge_dead_agents()
        self._update_ecosystem_metrics()

    def _process_smart_interactions(self):
        agent_list = list(self.agents.values())
        CONNECTION_COST = 0.0005

        for i, agent in enumerate(agent_list):
            if agent.health <= 0: continue

            for nid in list(agent.neighbors):
                neighbor = self.agents.get(nid)
                if not neighbor or neighbor.health <= 0:
                    agent.neighbors.remove(nid)
                    continue

                agent.health -= CONNECTION_COST
                compatibility = 1.0 - abs(agent.genome.C - neighbor.genome.C)
                if compatibility > 0.5:
                    agent.health = min(1.0, agent.health + 0.004 * compatibility)
                else:
                    agent.health -= 0.001

            if len(agent.neighbors) >= 6: continue

            for j, other in enumerate(agent_list[i+1:], i+1):
                if other.health <= 0 or len(other.neighbors) >= 6: continue
                dist = np.linalg.norm(agent.position - other.position)
                if dist < 4.0:
                    prediction, _ = agent.brain.evaluate_partner(other.genome, other.id)
                    if prediction + (random.random() - 0.5) * 0.2 > 0.0:
                        if other.id not in agent.neighbors:
                            agent.neighbors.append(other.id)
                            other.neighbors.append(agent.id)

    def _metabolic_feedback(self):
        for agent in self.agents.values():
            if agent.health <= 0: continue
            delta_e = agent.health - agent.prev_health
            if agent.neighbors and abs(delta_e) > 0.0001:
                share_delta = delta_e / len(agent.neighbors)
                for nid in agent.neighbors:
                    neighbor = self.agents.get(nid)
                    if neighbor:
                        agent.brain.learn_from_interaction(
                            neighbor.genome, neighbor.id, share_delta, self.state.time_step
                        )

    def _purge_dead_agents(self):
        dead_ids = [aid for aid, a in self.agents.items() if a.health <= 0.0]
        for aid in dead_ids:
            del self.agents[aid]

    def _update_ecosystem_metrics(self):
        if not self.agents: return
        total_health = sum(agent.health for agent in self.agents.values())
        self.state.total_energy = total_health / len(self.agents)
        total_connections = sum(len(agent.neighbors) for agent in self.agents.values())
        self.state.structure_coherence = total_connections / (len(self.agents) * 6 + 1e-6)

    def get_render_data(self):
        positions = []
        energies = []
        connections = []
        for aid, agent in self.agents.items():
            positions.append(agent.position.copy())
            energies.append(agent.health)
            connections.append(agent.neighbors.copy())
        return positions, energies, connections

    def inject_signal(self, position: np.ndarray, strength: float = 10.0):
        self._add_signal_source(position, strength)

    def clear_signals(self):
        self.signals.clear()
