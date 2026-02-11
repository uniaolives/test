"""
BIO-ARKHE: O Protocolo de Vida Digital.
Implementação dos 5 Princípios Biológicos de Inteligência.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
from .arkhe import ArkheGenome

# Constantes Fundamentais
MAX_NEIGHBORS = 6
SIGNAL_DECAY = 0.95
ASSEMBLY_THRESHOLD = 0.8

class MorphogeneticField:
    """Campo Morfogenético - O Meio Ambiente Inteligente"""

    def __init__(self, size: Tuple[int, int, int] = (100, 100, 100)):
        self.size = size
        self.signal_grid = np.zeros(size, dtype=np.float32)
        self.signal_history = []

    def update_field(self, agents: List['BioAgent']):
        """Atualiza o campo baseado na emissão (F) de todos os agentes."""
        self.signal_grid.fill(0)
        for agent in agents:
            if agent.health > 0:
                emission = agent.genome.F * agent.genome.E * agent.health
                pos = agent.position.astype(int)
                if np.all(pos >= 0) and np.all(pos < self.size):
                    self.signal_grid[tuple(pos)] += emission
        self._diffuse_signal()

    def get_local_gradient(self, position: np.ndarray) -> np.ndarray:
        x, y, z = position.astype(int)
        x = max(1, min(self.size[0] - 2, x))
        y = max(1, min(self.size[1] - 2, y))
        z = max(1, min(self.size[2] - 2, z))

        dx = (self.signal_grid[x+1, y, z] - self.signal_grid[x-1, y, z]) / 2.0
        dy = (self.signal_grid[x, y+1, z] - self.signal_grid[x, y-1, z]) / 2.0
        dz = (self.signal_grid[x, y, z+1] - self.signal_grid[x, y, z-1]) / 2.0

        gradient = np.array([dx, dy, dz], dtype=np.float32)
        norm = np.linalg.norm(gradient)
        return gradient / norm if norm > 1e-6 else np.zeros(3, dtype=np.float32)

    def get_gradient(self, pos: np.ndarray) -> np.ndarray:
        return self.get_local_gradient(pos)

    def get_signal_at(self, position: np.ndarray) -> float:
        x, y, z = position.astype(int)
        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]:
            return float(self.signal_grid[x, y, z])
        return 0.0

    def _diffuse_signal(self):
        new_grid = self.signal_grid * 0.8
        for axis in range(3):
            new_grid += 0.05 * np.roll(self.signal_grid, 1, axis=axis)
            new_grid += 0.05 * np.roll(self.signal_grid, -1, axis=axis)
        self.signal_grid = new_grid * SIGNAL_DECAY

class BioAgent:
    """Célula Autônoma com Cérebro Hebbiano Incorporado"""

    def __init__(self, id: int, position: np.ndarray, genome: ArkheGenome, velocity: np.ndarray = None):
        self.id = id
        self.position = position.astype(np.float32)
        self.velocity = velocity if velocity is not None else np.zeros(3, dtype=np.float32)
        self.genome = genome

        self.neighbors: List[int] = []
        self.health = 1.0
        self.age = 0
        self.prev_health = 1.0
        self.energy_reserve = 1.0

        from .constraint_engine import ConstraintLearner
        lr = 0.05 + (self.genome.I * 0.2)
        self.brain = ConstraintLearner(agent_id=id, learning_rate=lr)

        self.recent_interactions: List[Tuple[int, float]] = []
        self.memory_capacity = max(5, int(self.genome.I * 10))
        self.mood = "curious"
        self.last_action = "none"
        self.decision_reasoning = ""

    def perceive_environment(self, field: MorphogeneticField) -> Dict:
        signal = field.get_signal_at(self.position)
        gradient = field.get_local_gradient(self.position)
        return {
            'signal': signal,
            'gradient': gradient,
            'position': self.position.copy()
        }

    def sense_environment(self, field: MorphogeneticField) -> Dict:
        return self.perceive_environment(field)

    def decide_movement(self, sensory_data: Dict, other_agents: Dict[int, 'BioAgent']) -> np.ndarray:
        gradient = sensory_data['gradient']

        if self.mood == "social" or self.genome.C > 0.7:
            social_vector = self._calculate_social_vector(other_agents)
            if np.linalg.norm(social_vector) > 0.1:
                combined = gradient * 0.3 + social_vector * 0.7
                norm = np.linalg.norm(combined)
                if norm > 1e-6: combined /= norm
                self.last_action = "seeking_social"
                return combined * self.genome.E

        elif self.mood == "avoidant" or self.health < 0.3:
            avoid_vector = self._calculate_avoidance_vector(other_agents)
            if np.linalg.norm(avoid_vector) > 0.1:
                self.last_action = "avoiding"
                return avoid_vector * self.genome.E

        if np.linalg.norm(gradient) > 0.1:
            self.last_action = "following_gradient"
            return gradient * self.genome.E
        else:
            random_dir = np.random.randn(3).astype(np.float32)
            norm = np.linalg.norm(random_dir)
            if norm > 1e-6: random_dir /= norm
            self.last_action = "exploring"
            return random_dir * self.genome.E * 0.5

    def decide_action(self, sensory_data: Dict, other_agents: Dict[int, 'BioAgent']) -> np.ndarray:
        return self.decide_movement(sensory_data, other_agents)

    def _calculate_social_vector(self, other_agents: Dict[int, 'BioAgent']) -> np.ndarray:
        social_vector = np.zeros(3, dtype=np.float32)
        count = 0
        for other_id, other in list(other_agents.items())[:20]:
            if other_id == self.id or other.health <= 0: continue
            dist = np.linalg.norm(other.position - self.position)
            if dist < 20.0:
                score, _ = self.brain.evaluate_partner(other.genome, other_id)
                if score > 0.1:
                    social_vector += (other.position - self.position) / (dist + 1e-6)
                    count += 1
        return social_vector / count if count > 0 else social_vector

    def _calculate_avoidance_vector(self, other_agents: Dict[int, 'BioAgent']) -> np.ndarray:
        avoid_vector = np.zeros(3, dtype=np.float32)
        for other_id, other in list(other_agents.items())[:20]:
            if other_id == self.id or other.health <= 0: continue
            dist = np.linalg.norm(other.position - self.position)
            if dist < 10.0:
                score, _ = self.brain.evaluate_partner(other.genome, other_id)
                if score < -0.1:
                    avoid_vector += (self.position - other.position) / (dist + 1e-6)
        norm = np.linalg.norm(avoid_vector)
        return avoid_vector / norm if norm > 1e-6 else avoid_vector

    def evaluate_connection(self, partner: 'BioAgent') -> Tuple[bool, str]:
        score, reasoning = self.brain.evaluate_partner(partner.genome, partner.id)
        threshold = 0.2 if self.brain.successful_bonds > 10 else -0.3 if self.health < 0.4 else 0.0
        return score > threshold, f"Score: {score:.2f}, Threshold: {threshold:.2f}"

    def update_physics(self, dt: float):
        speed = np.linalg.norm(self.velocity)
        max_speed = self.genome.E * 3.0
        if speed > max_speed: self.velocity = self.velocity / speed * max_speed
        self.position += self.velocity * dt * 10.0
        self.position = np.clip(self.position, 0, 99)
        self.age += 1
        self.health -= 0.0005 * (1.0 - self.genome.E)
        if self.health > 0.8: self.mood = "social" if self.genome.C > 0.5 else "curious"
        elif self.health < 0.3: self.mood = "avoidant"
        self.energy_reserve = self.health * 0.8 + 0.2

    def update_state(self, action: np.ndarray, dt: float):
        self.velocity = self.velocity * 0.85 + action * 0.15
        self.update_physics(dt)

    def sense_and_act(self, field: MorphogeneticField, all_agents: Dict[int, 'BioAgent']):
        sensory = self.perceive_environment(field)
        action = self.decide_movement(sensory, all_agents)
        self.update_state(action, 0.016)

        if self.genome.C > 0.5 and len(self.neighbors) < MAX_NEIGHBORS:
            for other_id, other in all_agents.items():
                if other_id != self.id and other_id not in self.neighbors:
                    if np.linalg.norm(self.position - other.position) < 5.0:
                        accept_a, _ = self.evaluate_connection(other)
                        if accept_a and other.genome.C > 0.5:
                            self.neighbors.append(other_id)
                            other.neighbors.append(self.id)
                            break
