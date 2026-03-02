# arkhe/state_reconciler.py
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib

class ReconciliationStrategy(Enum):
    CONSENSUS = "consensus"      # Maioria vence
    UNION = "union"              # Merge de contextos
    PRIORITY = "priority"        # Provedor prioritário
    SMART_MERGE = "smart_merge" # ML-based reconciliation

@dataclass
class LLMState:
    provider: str
    content: Any
    context_hash: str
    timestamp: float
    confidence: float  # 0.0 a 1.0

class StateReconciler:
    """
    Implementa a identidade x² = x + 1 para estados paralelos:
    - x: estado individual de cada LLM
    - x²: estado reconciliado (emergente)
    - +1: contexto compartilhado (consistência)
    """

    def __init__(self, strategy: ReconciliationStrategy = ReconciliationStrategy.SMART_MERGE):
        self.strategy = strategy
        self.context_window: Dict[str, Any] = {}
        self.consistency_threshold = 0.87  # C do Arkhe

    async def reconcile(
        self,
        states: List[LLMState],
        document_id: str
    ) -> Dict[str, Any]:
        """
        Reconcilia múltiplos estados LLM em um estado consistente.
        """
        if not states:
            raise ValueError("No states to reconcile")

        # Calcular coerência global (C) e flutuação (F)
        C, F = self._calculate_consensus_metrics(states)

        if C < self.consistency_threshold:
            # Baixa coerência — ativar modo degradação
            return await self._degraded_reconciliation(states, document_id)

        # Estratégia baseada em consenso
        if self.strategy == ReconciliationStrategy.CONSENSUS:
            return self._consensus_merge(states, C, F)
        elif self.strategy == ReconciliationStrategy.SMART_MERGE:
            return await self._smart_merge(states, C, F)
        else:
            return self._priority_merge(states)

    def _calculate_consensus_metrics(self, states: List[LLMState]) -> tuple[float, float]:
        """Calcula C (coerência) e F (flutuação) dos estados."""
        if len(states) == 1:
            return 1.0, 0.0

        # Coerência = similaridade média entre pares
        similarities = []
        for i, s1 in enumerate(states):
            for s2 in states[i+1:]:
                sim = self._state_similarity(s1, s2)
                similarities.append(sim)

        C = sum(similarities) / len(similarities) if similarities else 0.0
        F = 1.0 - C  # Conservação C + F = 1

        return C, F

    def _state_similarity(self, s1: LLMState, s2: LLMState) -> float:
        """Calcula similaridade estrutural entre estados."""
        # Usar hash de conteúdo para similaridade rápida
        h1 = hashlib.sha256(str(s1.content).encode()).hexdigest()
        h2 = hashlib.sha256(str(s2.content).encode()).hexdigest()

        # Distância de Hamming normalizada (simplificada)
        diff = sum(c1 != c2 for c1, c2 in zip(h1, h2))
        return 1.0 - (diff / len(h1))

    async def _smart_merge(
        self,
        states: List[LLMState],
        C: float,
        F: float
    ) -> Dict[str, Any]:
        """
        Merge inteligente ponderado por confiança de cada provedor.
        """
        # Peso de cada estado = confiança * (1 - F_local)
        weighted_states = []
        for state in states:
            weight = state.confidence * (1.0 - F/len(states))
            weighted_states.append((state, weight))

        # Normalizar pesos
        total_weight = sum(w for _, w in weighted_states)
        if total_weight == 0:
            total_weight = 1.0

        # Construir estado reconciliado
        reconciled = {
            "content": self._weighted_merge_content(weighted_states),
            "metadata": {
                "providers": [s.provider for s in states],
                "coherence_C": C,
                "fluctuation_F": F,
                "strategy": "smart_merge",
                "satoshi": len(states) * 8.0  # bits de informação
            }
        }

        return reconciled

    def _weighted_merge_content(self, weighted_states: List[tuple]) -> Any:
        """Merge de conteúdo ponderado por confiança."""
        contents = [s.content for s, _ in weighted_states]
        weights = [w for _, w in weighted_states]

        if isinstance(contents[0], (int, float)):
            return sum(c * w for c, w in zip(contents, weights)) / sum(weights)

        # Para estruturas complexas, usar provedor de maior confiança
        max_idx = weights.index(max(weights))
        return contents[max_idx]

    async def _degraded_reconciliation(self, states: List[LLMState], document_id: str) -> Dict[str, Any]:
        """Tratamento para quando a coerência é baixa."""
        # Simplesmente pega o de maior confiança
        best_state = max(states, key=lambda s: s.confidence)
        return {
            "content": best_state.content,
            "metadata": {
                "document_id": document_id,
                "status": "degraded",
                "coherence_C": 0.5, # Nominal
                "fluctuation_F": 0.5
            }
        }

    def _priority_merge(self, states: List[LLMState]) -> Dict[str, Any]:
        # Implementação básica de prioridade
        return {"content": states[0].content, "metadata": {"strategy": "priority"}}

    def _consensus_merge(self, states: List[LLMState], C: float, F: float) -> Dict[str, Any]:
        # Implementação básica de consenso
        return {"content": states[0].content, "metadata": {"strategy": "consensus", "C": C, "F": F}}
