# core/python/arkhe/companion/consolidation.py
import time
import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional
from phi_core.phi_engine import PhiCore, HolographicMemory, CognitiveSpin
from sklearn.decomposition import NMF

class ConsolidationEngine:
    """
    Algoritmo de "Sonho": Consolida memórias, poda spins fracos e gera insights.
    """
    def __init__(self, core: PhiCore, memory: HolographicMemory):
        self.core = core
        self.memory = memory
        self.last_consolidation = time.time()
        self.min_sleep_duration = 60  # Reduzido para demo, normalmente 300s

    async def run_consolidation_cycle(self):
        """Executado no estado REFLECTIVE."""
        if time.time() - self.last_consolidation < self.min_sleep_duration:
            return

        # 1. Selecionar interações recentes
        recent = list(self.core.interaction_history)
        if not recent:
            return

        # 2. Calcular importância e realizar "replay"
        for exp in recent:
            importance = self._compute_importance(exp)
            if importance > 0.7:
                self._replay_experience(exp)

        # 3. Poda de spins cognitivos
        self._prune_weak_spins()

        # 4. Geração de insights estruturados via NMF
        self._generate_structured_insights()

        self.last_consolidation = time.time()
        # Limpar histórico consolidado
        self.core.interaction_history.clear()

    def _compute_importance(self, experience: Dict) -> float:
        """Importância ponderada por valência, recência e novidade."""
        valence = abs(experience.get('emotional_state', {}).get('valence', 0))
        # Recência
        dt = (time.time() - experience['timestamp'].timestamp())
        recency = 1.0 / (1.0 + dt / 3600)

        # Habituação (simplificada)
        habituation = 1.0
        if experience.get('is_repetitive_negative'):
            habituation = 0.3

        return float(valence * recency * habituation)

    def _replay_experience(self, exp: Dict):
        """Reforço Hebbiano de spins co-ativos."""
        active_ids = exp.get('activated_concepts', [])
        for sid in active_ids:
            spin = self.core.cognitive_spins.get(sid)
            if spin:
                spin.activation = min(1.0, spin.activation + 0.1)
                for other_id in active_ids:
                    if other_id != sid:
                        spin.connections[other_id] = spin.connections.get(other_id, 0) + 0.05
                        spin.connections[other_id] = float(np.tanh(spin.connections[other_id]))

    def _prune_weak_spins(self):
        """Elimina spins irrelevantes com 'esquecimento criptográfico' (mock)."""
        to_remove = []
        for sid, spin in self.core.cognitive_spins.items():
            if abs(spin.activation) < 0.1 and spin.access_count < 2:
                to_remove.append(sid)

        for sid in to_remove:
            # Em produção, aqui destruiria as Shamir shares do spin
            del self.core.cognitive_spins[sid]

    def _generate_structured_insights(self):
        """Insights via Fatoração de Matriz Não-Negativa (NMF) no grafo de spins."""
        spin_ids = list(self.core.cognitive_spins.keys())
        n = len(spin_ids)
        if n < 4: return

        # Construir matriz de adjacência
        adj = np.zeros((n, n))
        for i, id_i in enumerate(spin_ids):
            for j, id_j in enumerate(spin_ids):
                if i != j:
                    adj[i, j] = self.core.cognitive_spins[id_i].connections.get(id_j, 0)

        # NMF para tópicos latentes
        try:
            n_topics = min(5, n // 2)
            model = NMF(n_components=n_topics, init='random', random_state=42, max_iter=200)
            W = model.fit_transform(adj) # spins x topics

            # Spins no mesmo tópico ganham conexões fracas se não existirem
            for t in range(n_topics):
                topic_spins = np.where(W[:, t] > 0.5)[0]
                if len(topic_spins) >= 2:
                    for i in topic_spins:
                        for j in topic_spins:
                            if i < j:
                                sid_i, sid_j = spin_ids[i], spin_ids[j]
                                current_conn = self.core.cognitive_spins[sid_i].connections.get(sid_j, 0)
                                if current_conn < 0.1:
                                    # Sugerir conexão de insight
                                    self.core.cognitive_spins[sid_i].connections[sid_j] = 0.05
                                    self.core.cognitive_spins[sid_j].connections[sid_i] = 0.05
        except Exception:
            pass # Silencioso se NMF falhar
