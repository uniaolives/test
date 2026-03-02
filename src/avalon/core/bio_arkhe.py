"""
BIO-ARKHE v3.0 - DNA Quaternário e Campo Morfogenético
Implementação standalone sem dependências externas
BIO-ARKHE: O Protocolo de Vida Digital.
Implementação dos 5 Princípios Biológicos de Inteligência.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ArkheGenome:
    """
    DNA digital com 4 dimensões fundamentais:
    C: Chemistry (Química) - Compatibilidade estrutural
    I: Information (Informação) - Capacidade de processamento
    E: Energy (Energia) - Metabolismo e velocidade
    F: Function (Função) - Especialização e sinalização
    """
    C: float
    I: float
    E: float
    F: float

    def to_vector(self) -> np.ndarray:
        """Converte genoma para vetor numpy."""
        return np.array([self.C, self.I, self.E, self.F], dtype=np.float32)

    def mutate(self, rate: float = 0.1) -> 'ArkheGenome':
        """Cria mutação gaussiana do genoma."""
        def clamp(x):
            return max(0.05, min(0.95, x))

        return ArkheGenome(
            C=clamp(self.C + np.random.normal(0, rate)),
            I=clamp(self.I + np.random.normal(0, rate)),
            E=clamp(self.E + np.random.normal(0, rate)),
            F=clamp(self.F + np.random.normal(0, rate))
        )


class MorphogeneticField:
    """
    Campo escalar 3D que permeia o espaço da simulação.
    Implementa difusão de sinais metabólicos e gradientes químicos.
    """

    def __init__(self, size: Tuple[int, int, int] = (100, 100, 100)):
        self.size = size
        self.grid = np.zeros(size, dtype=np.float32)

        # Constantes de difusão
        self.decay_rate = 0.96
        self.diffusion_rate = 0.1

    def add_signal(self, x: float, y: float, z: float, strength: float) -> None:
        """Adiciona sinal em coordenadas específicas."""
        ix, iy, iz = int(x), int(y), int(z)

        if (0 <= ix < self.size[0] and
            0 <= iy < self.size[1] and
            0 <= iz < self.size[2]):
            self.grid[ix, iy, iz] += strength

    def get_signal_at(self, x: float, y: float, z: float) -> float:
        """Retorna intensidade do sinal em posição."""
        ix, iy, iz = int(x), int(y), int(z)

        if (0 <= ix < self.size[0] and
            0 <= iy < self.size[1] and
            0 <= iz < self.size[2]):
            return float(self.grid[ix, iy, iz])
        return 0.0

    def get_gradient(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Calcula gradiente do campo por diferenças finitas.
        Usado para quimiotaxia (movimento em direção a sinais).
        """
        ix, iy, iz = int(x), int(y), int(z)

        # Garante limites para cálculo do gradiente
        ix = max(1, min(self.size[0] - 2, ix))
        iy = max(1, min(self.size[1] - 2, iy))
        iz = max(1, min(self.size[2] - 2, iz))

        grad = np.array([
            (self.grid[ix + 1, iy, iz] - self.grid[ix - 1, iy, iz]) / 2.0,
            (self.grid[ix, iy + 1, iz] - self.grid[ix, iy - 1, iz]) / 2.0,
            (self.grid[ix, iy, iz + 1] - self.grid[ix, iy, iz - 1]) / 2.0
        ], dtype=np.float32)

        # Normaliza
        norm = np.linalg.norm(grad)
        if norm > 1e-6:
            return grad / norm
        return np.random.randn(3).astype(np.float32) * 0.1

    def diffuse_and_decay(self) -> None:
        """
        Aplica difusão isotrópica e decaimento temporal.
        Implementação vetorizada usando numpy (sem scipy).
        """
        # Difusão via rolling - espalha para 6 vizinhos ortogonais
        neighbors = (
            np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) +
            np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1) +
            np.roll(self.grid, 1, axis=2) + np.roll(self.grid, -1, axis=2)
        )

        # Atualização: conservação + difusão + decaimento
        self.grid = (
            self.grid * (1 - 6 * self.diffusion_rate) +
            neighbors * self.diffusion_rate
        ) * self.decay_rate


class BioAgent:
    """
    Agente autônomo com corpo físico e cognição embarcada.
    """

    def __init__(self, agent_id: int, x: float, y: float, z: float,
                 genome: ArkheGenome):
        self.id = agent_id

        # Estado físico
        self.position = np.array([x, y, z], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.genome = genome

        # Metabolismo
        self.health = 0.7 + genome.E * 0.3  # Energia inicial baseada no gene E
        self.age = 0.0
        self.alive = True

        # Conectividade social (máximo 6 conexões - número de coordenação)
        self.connections: list = []
        self.bond_strengths: dict = {}

        # Cérebro (inicializado externamente)
        self.brain = None

        # Estado comportamental
        self.state = "exploring"
        self.last_decision = ""

    def set_brain(self, brain) -> None:
        """Conecta o sistema cognitivo."""
        self.brain = brain

    def is_alive(self) -> bool:
        return self.alive and self.health > 0

    def get_position(self) -> Tuple[float, float, float]:
        return (float(self.position[0]), float(self.position[1]), float(self.position[2]))

    def apply_physics(self, dt: float, field_size: Tuple[int, ...]) -> None:
        """
        Atualiza física do agente com:
        - Integração de velocidade
        - Fricção viscosa
        - Metabolismo basal
        - Condições de contorno (bounce)
        """
        # Atualiza posição
        self.position += self.velocity * dt

        # Fricção (arrasto do meio)
        self.velocity *= 0.92

        # Metabolismo: custo de movimento + manutenção basal
        speed = np.linalg.norm(self.velocity)
        movement_cost = float(speed * speed * 0.001)  # Custo quadrático
        base_cost = 0.0005 * (1.1 - self.genome.E)
        self.health -= (movement_cost + base_cost) * dt

        # Envelhecimento
        self.age += dt

        # Condições de contorno - reflexão suave nas bordas
        for i, (pos, limit) in enumerate(zip(self.position, field_size)):
            if pos <= 0:
                self.position[i] = 0.1
                self.velocity[i] = abs(self.velocity[i]) * 0.5
            elif pos >= limit - 1:
                self.position[i] = limit - 1.1
                self.velocity[i] = -abs(self.velocity[i]) * 0.5

        # Morte
        if self.health <= 0 or self.age > 1000:
            self.alive = False

    def form_bond(self, other_agent, strength: float = 0.5) -> bool:
        """
        Tenta formar conexão simbiótica com outro agente.
        Retorna True se a conexão foi estabelecida.
        """
        # Verifica limites de conectividade (máximo 6 vizinhos)
        if (len(self.connections) >= 6 or
            len(other_agent.connections) >= 6):
            return False

        if other_agent.id not in self.connections:
            self.connections.append(other_agent.id)
            self.bond_strengths[other_agent.id] = strength

            # Conexão recíproca
            if self.id not in other_agent.connections:
                other_agent.connections.append(self.id)
                other_agent.bond_strengths[self.id] = strength

            return True
        return False

    def break_bond(self, other_id: int) -> None:
        """Rompe conexão com outro agente."""
        if other_id in self.connections:
            self.connections.remove(other_id)
            self.bond_strengths.pop(other_id, None)
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
