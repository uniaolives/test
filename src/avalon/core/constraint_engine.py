"""
CONSTRAINT ENGINE v3.0 - Micro-cérebro Hebbiano com Memória Temporal
Implementa aprendizado por reforço metabólico e curiosidade artificial
"""

import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SynapticTrace:
    """Traço de memória episódica com decaimento temporal"""
    partner_signature: str
    energy_delta: float
    timestamp: float

    def decay_factor(self, current_time: float, tau: float = 100.0) -> float:
        """Peso da memória decai exponencialmente com o tempo"""
        return np.exp(-abs(current_time - self.timestamp) / tau)


class ConstraintLearner:
    """
    Sistema cognitivo Hebbiano com:
    - Plasticidade sináptica dependente do tempo (STDP)
    - Exploração guiada por curiosidade (novelty search)
    - Memória de trabalho para reconhecimento de padrões sequenciais
    - Meta-aprendizado adaptativo
    """

    def __init__(self, agent_id: int, genome_vector: Optional[np.ndarray] = None):
        self.agent_id = agent_id

        # Pesos sinápticos para [C, I, E, F] - inicialmente tabula rasa
        self.weights = np.zeros(4, dtype=np.float32)
        self.bias = 0.0
        self.learning_rate = 0.15

        # Memória de trabalho (últimas 15 interações)
        self.working_memory: deque[SynapticTrace] = deque(maxlen=15)

        # Estado de exploração-curiosidade
        self.exploration_rate = 0.3
        self.novelty_threshold = 0.2

        # Estatísticas para perfil cognitivo
        self.metrics = {
            'successful_interactions': 0,
            'failed_interactions': 0,
            'total_energy_gained': 0.0,
            'total_energy_lost': 0.0,
            'prediction_errors': []
        }

        # Autoconhecimento: inicialização baseada no próprio genoma
        if genome_vector is not None:
            # Preferência inicial por similaridade (clones de si mesmo)
            self.weights = genome_vector * 0.3

    def evaluate_partner(self, partner_genome, current_time: float = 0.0) -> Tuple[float, str]:
        """
        Avalia um parceiro potencial usando integração bayesiana aproximada.

        Combina:
        1. Predição baseada em pesos sinápticos (memória semântica)
        2. Recuperação de memória episódica específica
        3. Componente de exploração-curiosidade

        Returns:
            score: Valor entre -1.0 (evitar) e 1.0 (aproximar)
            reasoning: String explicativa da decisão
        """
        features = np.array([
            partner_genome.C,
            partner_genome.I,
            partner_genome.E,
            partner_genome.F
        ], dtype=np.float32)

        # 1. Predição baseada em conhecimento generalizado (pesos)
        raw_prediction = np.dot(self.weights, features) + self.bias
        semantic_score = np.tanh(raw_prediction)

        # 2. Busca em memória episódica
        memory_score = self._query_memory(features, current_time)

        # 3. Integração bayesiana aproximada
        if memory_score is not None:
            # Memória específica tem peso maior (0.7) se disponível
            final_score = 0.7 * memory_score + 0.3 * semantic_score
            reasoning = f"Memória({memory_score:+.2f}) + Intuição({semantic_score:+.2f})"
        else:
            final_score = semantic_score
            reasoning = f"Intuição({semantic_score:+.2f})"

        # 4. Modulação por curiosidade (novelty search)
        # Agentes inexperientes exploram mais
        uncertainty = 1.0 - min(1.0, np.mean(np.abs(self.weights)) * 2)
        if random.random() < self.exploration_rate * uncertainty:
            noise = (random.random() - 0.5) * 0.5
            final_score += noise
            reasoning += f" [Exploração{noise:+.2f}]"

        return float(np.clip(final_score, -1.0, 1.0)), reasoning

    def _query_memory(self, features: np.ndarray, current_time: float) -> Optional[float]:
        """
        Consulta memória episódica por experiências similares.
        Usa decaimento temporal - memórias recentes têm mais peso.
        """
        if not self.working_memory:
            return None

        best_match = None
        best_score = -float('inf')

        for trace in self.working_memory:
            try:
                # Decodifica assinatura do genoma
                parts = trace.partner_signature.split('_')
                if len(parts) == 4:
                    trace_features = np.array([float(p) for p in parts], dtype=np.float32)

                    # Similaridade por distância euclidiana normalizada
                    distance = np.linalg.norm(features - trace_features)
                    similarity = max(0.0, 1.0 - distance / 2.0)

                    if similarity > 0.75:  # Threshold de similaridade
                        # Peso pelo decaimento temporal
                        decay = trace.decay_factor(current_time)
                        weighted_value = trace.energy_delta * 5 * decay * similarity

                        if weighted_value > best_score:
                            best_score = weighted_value
                            best_match = trace
            except (ValueError, IndexError):
                continue

        if best_match:
            return float(np.clip(best_score, -1.0, 1.0))
        return None

    def learn_from_experience(self, partner_genome, energy_delta: float,
                             current_time: float = 0.0) -> None:
        """
        Atualiza pesos sinápticos via regra Hebbiana modificada.

        Implementa TD-learning simplificado:
        - Reforço positivo (LTP) quando energia aumenta
        - Inibição (LTD) quando energia diminui
        - Magnitude do aprendizado proporcional à surpresa (erro de predição)
        """
        features = np.array([
            partner_genome.C,
            partner_genome.I,
            partner_genome.E,
            partner_genome.F
        ], dtype=np.float32)

        # Calcula erro de predição (surpresa)
        prev_prediction = np.tanh(np.dot(self.weights, features) + self.bias)
        observed_outcome = np.clip(energy_delta * 5, -1.0, 1.0)
        prediction_error = observed_outcome - prev_prediction

        self.metrics['prediction_errors'].append(abs(prediction_error))
        if len(self.metrics['prediction_errors']) > 20:
            self.metrics['prediction_errors'].pop(0)

        # Força do aprendizado = taxa_base * |erro| (surpresa)
        surprise_factor = min(abs(prediction_error) * 2, 1.0)
        effective_lr = self.learning_rate * surprise_factor

        # Atualização dos pesos (Regra Delta)
        if energy_delta > 0:
            # Long-term potentiation (LTP)
            self.weights += effective_lr * features
            self.bias += effective_lr * 0.3
            self.metrics['successful_interactions'] += 1
            self.metrics['total_energy_gained'] += energy_delta

            # Sucesso reduz exploração (exploitation)
            self.exploration_rate *= 0.98
        else:
            # Long-term depression (LTD)
            self.weights -= effective_lr * features * 0.5
            self.bias -= effective_lr * 0.15
            self.metrics['failed_interactions'] += 1
            self.metrics['total_energy_lost'] += abs(energy_delta)

            # Fracasso aumenta exploração
            self.exploration_rate = min(self.exploration_rate * 1.03, 0.6)

        # Homeostase sináptica - evita explosão de pesos
        weight_norm = np.linalg.norm(self.weights)
        if weight_norm > 2.5:
            self.weights = (self.weights / weight_norm) * 2.5

        self.weights = np.clip(self.weights, -2.5, 2.5)
        self.bias = np.clip(self.bias, -1.5, 1.5)

        # Armazena na memória de trabalho
        genome_hash = (f"{partner_genome.C:.3f}_{partner_genome.I:.3f}_"
                      f"{partner_genome.E:.3f}_{partner_genome.F:.3f}")

        self.working_memory.append(SynapticTrace(
            partner_signature=genome_hash,
            energy_delta=energy_delta,
            timestamp=current_time
        ))

    def get_cognitive_profile(self) -> str:
        """
        Classifica o agente baseado em sua história de interações.
        """
        total = (self.metrics['successful_interactions'] +
                self.metrics['failed_interactions'])

        if total < 5:
            return "Neófito"

        success_rate = self.metrics['successful_interactions'] / total

        if success_rate > 0.75:
            return "Especialista"
        elif success_rate > 0.45:
            return "Aprendiz"
        elif success_rate > 0.25:
            return "Explorador"
        else:
            return "Cauteloso"

    def get_preferences(self) -> str:
        """
        Descreve preferências aprendidas baseadas nos pesos dominantes.
        """
        labels = ["Química", "Informação", "Energia", "Função"]
        max_idx = np.argmax(np.abs(self.weights))

        if abs(self.weights[max_idx]) < 0.15:
            return "Explorando padrões"

        direction = "atraído por" if self.weights[max_idx] > 0 else "evita"
        return f"{direction} {labels[max_idx]} ({abs(self.weights[max_idx]):.2f})"

    def get_cognitive_state(self) -> dict:
        """Retorna estado cognitivo completo para visualização."""
        total = (self.metrics['successful_interactions'] +
                self.metrics['failed_interactions'])

        return {
            'profile': self.get_cognitive_profile(),
            'preferences': self.get_preferences(),
            'exploration_rate': round(self.exploration_rate, 3),
            'learning_rate': round(self.learning_rate, 3),
            'memory_size': len(self.working_memory),
            'total_interactions': total,
            'success_rate': (round(self.metrics['successful_interactions'] / max(1, total), 2)),
            'avg_prediction_error': (round(np.mean(self.metrics['prediction_errors']), 3)
                                   if self.metrics['prediction_errors'] else 0),
            'weights': [round(w, 3) for w in self.weights.tolist()],
            'bias': round(self.bias, 3)
        }
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
