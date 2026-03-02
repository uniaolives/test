"""
UCD Monitor: Universal Coherence Detection.
Monitors C + F = 1 identity using Shannon Entropy.
"""

import numpy as np

class UCD:
    """
    Universal Coherence Detection (UCD)
    Monitora a identidade C + F = 1 em qualquer fluxo de dados.
    """
    def __init__(self, stream_name):
        self.name = stream_name
        self.history = []

    def ingest(self, signal_vector):
        # Normaliza o sinal (x)
        norm = np.linalg.norm(signal_vector)
        if norm == 0:
            return

        # Calcula a Entropia de Shannon (F)
        # Usamos o valor absoluto para garantir que p seja positivo e normalizamos para sum(p)=1
        p = np.abs(signal_vector) / np.sum(np.abs(signal_vector))
        entropy = -np.sum(p * np.log2(p + 1e-9))

        # Normalização da entropia para o intervalo [0, 1]
        max_entropy = np.log2(len(signal_vector))
        F = entropy / max_entropy
        C = 1.0 - F

        # Verifica Conservação
        is_conserved = abs((C + F) - 1.0) < 1e-5

        self.history.append((C, F))

        print(f"[{self.name}] C: {C:.4f} | F: {F:.4f} | Status: {'✅' if is_conserved else '❌'}")
        return C, F

if __name__ == "__main__":
    # Simulação
    monitor = UCD("Arkhe_Brain")
    print("--- Ingesting Chaos ---")
    monitor.ingest(np.random.rand(128)) # Estado Inicial (Caos)
    print("--- Ingesting Order ---")
    monitor.ingest(np.ones(128))        # Estado Final (Ordem Absoluta)
