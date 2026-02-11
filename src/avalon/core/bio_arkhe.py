"""
BIO-ARKHE: Active Component Assembly Architecture.
Implementação dos 5 Princípios Biológicos de Inteligência.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Constantes de Vida
MAX_NEIGHBORS = 6  # Simetria Hexagonal (Packing eficiente)
SIGNAL_DECAY = 0.95 # O sinal enfraquece com a distância
ASSEMBLY_THRESHOLD = 0.1 # Afinidade necessária para ligação (reduzido para teste)

@dataclass
class ArkheGenome:
    """O DNA do Agente: Define sua personalidade e função."""
    C: float  # Chemistry: Força de ligação (0.0 - 1.0)
    I: float  # Information: Capacidade de processamento (armazenamento de restrições)
    E: float  # Energy: Mobilidade e metabolismo
    F: float  # Function: Intensidade de sinalização (comunicação)

class MorphogeneticField:
    """
    O Meio Ambiente Ativo.
    Mantém o mapa de 'cheiros' (sinais) que guia os agentes.
    """
    def __init__(self, size: Tuple[int, int, int] = (20, 20, 20)):
        self.size = size
        # Grid escalar para sinalização
        self.signal_grid = np.zeros(size)

    def update_field(self, agents: List['BioAgent']):
        """
        Atualiza o campo baseado na emissão (F) de todos os agentes.
        Simula difusão e decaimento.
        """
        # Decaimento natural (entropia)
        self.signal_grid *= SIGNAL_DECAY

        # Agentes emitem sinal na sua posição
        for agent in agents:
            # Scale agent position to grid size
            # Assuming agent positions are roughly in range [-10, 10]
            grid_pos = ((agent.position + 10) / 20 * (np.array(self.size) - 1)).astype(int)

            # Verifica limites
            if np.all(grid_pos >= 0) and np.all(grid_pos < self.size):
                # A força do sinal é baseada na Função (F) e Energia (E)
                emission = agent.genome.F * agent.genome.E
                self.signal_grid[tuple(grid_pos)] += emission

    def get_gradient(self, pos: np.ndarray) -> np.ndarray:
        """Retorna o vetor direção para o sinal mais forte."""
        grid_pos = ((pos + 10) / 20 * (np.array(self.size) - 1)).astype(int)

        if not (np.all(grid_pos >= 0) and np.all(grid_pos < self.size)):
            return np.random.randn(3) * 0.1

        # Simplified 3D gradient check in 6 directions
        max_signal = self.signal_grid[tuple(grid_pos)]
        best_dir = np.zeros(3)

        for axis in range(3):
            for delta in [-1, 1]:
                neighbor_pos = grid_pos.copy()
                neighbor_pos[axis] += delta
                if 0 <= neighbor_pos[axis] < self.size[axis]:
                    signal = self.signal_grid[tuple(neighbor_pos)]
                    if signal > max_signal:
                        max_signal = signal
                        best_dir = np.zeros(3)
                        best_dir[axis] = delta

        return best_dir * 0.5 + np.random.randn(3) * 0.05

class BioAgent:
    """
    A Célula Autônoma.
    Princípio 1: Multiscale Autonomy
    """
    def __init__(self, id: int, position: np.ndarray, genome: ArkheGenome):
        self.id = id
        self.position = position
        self.velocity = np.zeros(3)
        self.genome = genome

        # Estado interno
        self.neighbors: List[int] = [] # IDs dos vizinhos conectados
        self.health = 1.0

    def sense_and_act(self, field: MorphogeneticField, all_agents: Dict[int, 'BioAgent']):
        """
        Ciclo de Vida do Agente.
        Princípio 5: Pervasive Signaling
        Princípio 2: Growth via Self-Assembly
        """
        # 1. PERCEPÇÃO (Ler o campo)
        gradient = field.get_gradient(self.position)

        # 2. DECISÃO (Arkhe Logic)
        # Movimento guiado pelo campo (Gradiente) e Energia (E)
        desired_velocity = gradient * self.genome.E

        # 3. AUTO-MONTAGEM (Self-Assembly)
        if self.genome.C > 0.5 and len(self.neighbors) < MAX_NEIGHBORS:
            self._try_bond(all_agents)

        # 4. EXECUÇÃO FÍSICA
        if self.neighbors:
            # Cohesion and alignment with neighbors
            neighbor_pos = np.mean([all_agents[nid].position for nid in self.neighbors], axis=0)
            pull = (neighbor_pos - self.position) * 0.1
            self.velocity += pull + desired_velocity * 0.5
        else:
            self.velocity += desired_velocity

        # Atualiza física
        self.position += self.velocity
        self.velocity *= 0.85 # Atrito aumentado

        # Boundary check
        self.position = np.clip(self.position, -10, 10)

    def _try_bond(self, all_agents: Dict[int, 'BioAgent']):
        """Tenta se conectar a vizinhos próximos baseados em afinidade."""
        for other_id, other in all_agents.items():
            if other_id == self.id or other_id in self.neighbors:
                continue

            dist = np.linalg.norm(self.position - other.position)
            if dist < 1.0 and len(other.neighbors) < MAX_NEIGHBORS:
                # Afinidade baseada na proximidade do genoma (I - Information)
                affinity = 1.0 - abs(self.genome.I - other.genome.I)
                if affinity > ASSEMBLY_THRESHOLD:
                    self.neighbors.append(other_id)
                    other.neighbors.append(self.id)
                    break
