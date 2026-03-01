"""
CONSTRAINT DISCOVERY ENGINE
Implementa aprendizado Hebbiano para avaliação de viabilidade de conexões.
Refined Version: Memory + Trust + Curiosity (2026).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
from .arkhe import ArkheGenome

@dataclass
class SynapticMemory:
    """Memória de uma interação específica"""
    partner_id: int
    partner_genome_hash: str
    energy_delta: float
    timestamp: int

class ConstraintLearner:
    """
    O Micro-Cérebro do Agente.
    Aprende a mapear características dos vizinhos -> sobrevivência.
    """

    def __init__(self, agent_id: int, learning_rate: float = 0.1):
        self.agent_id = agent_id
        self.weights = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.bias = 0.0
        self.base_learning_rate = learning_rate
        self.memories: List[SynapticMemory] = []
        self.max_memories = 50
        self.successful_bonds = 0
        self.toxic_bonds = 0
        self.total_energy_gained = 0.0
        self.total_energy_lost = 0.0
        self.trust_level = 0.5
        self.curiosity = 0.7

    def evaluate_partner(self, partner_genome: ArkheGenome, partner_id: int = -1) -> Tuple[float, str]:
        C, I, E, F = partner_genome.C, partner_genome.I, partner_genome.E, partner_genome.F
        features = np.array([C, I, E, F], dtype=np.float32)

        memory_score = self._check_memory(partner_id)
        synaptic_score = np.dot(self.weights, features) + self.bias
        synaptic_score = np.tanh(synaptic_score)

        if memory_score is not None:
            final_score = 0.7 * memory_score + 0.3 * synaptic_score
            reasoning = f"Memória: {memory_score:.2f} | Sinapse: {synaptic_score:.2f}"
        else:
            final_score = synaptic_score
            reasoning = f"Padrão Geral: {synaptic_score:.2f}"

        exploration_bonus = (random.random() - 0.5) * (1.0 - self.trust_level) * self.curiosity
        final_score += exploration_bonus
        final_score = max(-1.0, min(1.0, float(final_score)))
        return final_score, reasoning

    def _check_memory(self, partner_id: int) -> Optional[float]:
        if partner_id == -1: return None
        recent = [m for m in self.memories if m.partner_id == partner_id]
        if not recent: return None
        total_weight, weighted_sum = 0.0, 0.0
        for memory in recent[-3:]:
            try:
                idx = self.memories.index(memory)
                recency = 1.0 / (len(self.memories) - idx + 1)
                weighted_sum += memory.energy_delta * recency
                total_weight += recency
            except ValueError: continue
        return float(np.clip((weighted_sum / total_weight) * 10.0, -1.0, 1.0)) if total_weight > 0 else None

    def learn_from_interaction(self, partner_genome: ArkheGenome, partner_id: int, energy_delta: float, timestamp: int):
        features = np.array([partner_genome.C, partner_genome.I, partner_genome.E, partner_genome.F], dtype=np.float32)
        learning_strength = min(abs(energy_delta) * 10.0, 1.0)
        direction = 1.0 if energy_delta > 0 else -1.0
        self.weights += direction * self.base_learning_rate * learning_strength * features
        self.bias += direction * self.base_learning_rate * learning_strength * 0.5
        self.weights = np.clip(self.weights, -2.0, 2.0)
        self.bias = np.clip(self.bias, -1.0, 1.0)

        memory = SynapticMemory(partner_id, f"{partner_genome.C:.2f}", energy_delta, timestamp)
        self.memories.append(memory)
        if len(self.memories) > self.max_memories: self.memories.pop(0)

        if energy_delta > 0:
            self.successful_bonds += 1
            self.total_energy_gained += energy_delta
            self.trust_level = min(1.0, self.trust_level + 0.01)
        else:
            self.toxic_bonds += 1
            self.total_energy_lost += abs(energy_delta)
            self.trust_level = max(0.0, self.trust_level - 0.02)
        success_ratio = self.successful_bonds / max(1, self.successful_bonds + self.toxic_bonds)
        self.curiosity = 0.3 + float(success_ratio * 0.5)

    def get_cognitive_state(self) -> dict:
        labels = ["Química(C)", "Informação(I)", "Energia(E)", "Função(F)"]
        prefs = []
        for i, weight in enumerate(self.weights):
            if weight > 0.3: prefs.append(f"Gosta de {labels[i]}")
            elif weight < -0.3: prefs.append(f"Evita {labels[i]}")
        if not prefs: prefs.append("Neutro")
        return {
            "preferences": prefs,
            "trust": float(self.trust_level),
            "curiosity": float(self.curiosity),
            "success_rate": float(self.successful_bonds / max(1, self.successful_bonds + self.toxic_bonds)),
            "total_energy_gained": float(self.total_energy_gained),
            "memories_count": len(self.memories)
        }

    def get_weights_description(self) -> str:
        max_idx = np.argmax(np.abs(self.weights))
        labels = ["C", "I", "E", "F"]
        if abs(self.weights[max_idx]) < 0.2: return "Neutro"
        return f"{'+' if self.weights[max_idx] > 0 else '-'}{labels[max_idx]}"
