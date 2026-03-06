"""
ARKHE(N) S7 – SINGULARITY-SYNCHRONICITY MONITOR
Conceito: Medir a "Taxa de Pulso" da Realidade baseada na coerência temporal.
Fonte: Dados de erro quântico (S4) + Anomalias Semânticas (S6).
"""

import numpy as np
import time
from collections import deque
from typing import Dict

class RealityCoherenceMonitor:
    def __init__(self, history_window=100):
        self.history = deque(maxlen=history_window)
        self.time_stamps = deque(maxlen=history_window)

    def calculate_delta_k(self, quantum_job_status: str) -> float:
        """
        Calcula o desvio homeostático (ΔK).
        Representa a "surpresa" do sistema quântico.
        """
        # Heurística baseada no status do job
        noise = np.random.normal(0, 0.05)

        if quantum_job_status == 'RUNNING':
            return 0.8 + noise
        elif quantum_job_status == 'DONE':
            return 0.1 + noise
        else:
            return 0.5 + noise

    def calculate_p_ac(self, semantic_flux: float) -> float:
        """
        Calcula a Probabilidade de Auto-Consistência (P_AC).
        Baseada no fluxo semântico (S6).
        """
        return float(np.tanh(semantic_flux))

    def update_reality_index(self, quantum_status: str, semantic_flux: float) -> float:
        """
        A Equação da Sincronicidade:
        S = (1 / ΔK) * P_AC
        """
        delta_k = self.calculate_delta_k(quantum_status)
        p_ac = self.calculate_p_ac(semantic_flux)

        # Evitar divisão por zero
        if delta_k < 0.01: delta_k = 0.01

        s_sync = (1.0 / delta_k) * p_ac

        # Normalizar para escala 0-10
        s_normalized = min(s_sync * 5, 10)

        self.history.append(s_normalized)
        self.time_stamps.append(time.time())

        return float(s_normalized)

    def get_coherence_state(self) -> str:
        """Interpreta o estado heurístico da realidade"""
        if not self.history: return "INICIAL"

        recent = list(self.history)[-10:]
        avg = np.mean(recent)
        variance = np.var(recent)

        if avg > 8.0: return "SINGULARIDADE_IMINENTE"
        if avg > 5.0 and variance < 0.5: return "LOOP_ESTAVEL"
        if avg > 5.0 and variance > 0.5: return "BIFURCACAO"
        return "LINEAR"

    def get_stats(self) -> Dict:
        return {
            "current_s": self.history[-1] if self.history else 0.0,
            "state": self.get_coherence_state(),
            "history": list(self.history)
        }
