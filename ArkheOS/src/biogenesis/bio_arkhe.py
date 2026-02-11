"""
BIO-ARKHE v3.0 - DNA Quaternário e Campo Morfogenético
Implementação standalone sem dependências externas
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict


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
        self.last_action = ""

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
