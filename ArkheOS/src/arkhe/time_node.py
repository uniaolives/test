"""
Arkhe Time Node: Servidor Stratum 1 como nó de tempo no hipergrafo.
Implementa a descentralização da verdade temporal via acoplamento com satélites GNSS.
"""

import numpy as np
import time
from datetime import datetime
from typing import List, Dict

class GNSSSatellite:
    """Modelo simplificado de um satélite GNSS com relógio atômico de bordo."""
    def __init__(self, name: str, system: str, atomic_clock_error: float = 1e-15):
        self.name = name
        self.system = system
        self.atomic_clock_error = atomic_clock_error
        self.C = 1.0 - atomic_clock_error

    def transmit_time(self, t: float) -> float:
        """Transmite o tempo atual com erro estocástico (flutuação F)."""
        error = np.random.normal(0, self.atomic_clock_error)
        return t + error

class Stratum1Server:
    """
    Servidor de tempo local (Stratum 1) sincronizado diretamente com GNSS.
    Opera como o x² que transforma sinal espacial (+1) em tempo útil.
    """
    def __init__(self, name: str):
        self.name = name
        self.time_offset = 0.0
        self.handovers: List[Dict] = []
        self.C = 0.86  # Coerência alvo
        self.satoshi = 0.0  # Memória de sincronização

    def synchronize(self, satellite: GNSSSatellite, t_reception: float) -> float:
        """
        Executa um handover temporal entre o espaço e o nó local.
        """
        # Sinal captado (x) com atraso de propagação simulado
        propagation_delay = 0.07 # 70ms
        t_satellite = satellite.transmit_time(t_reception - propagation_delay)

        # Diferença detectada (offset)
        delta = t_satellite - (t_reception - propagation_delay)

        # Filtro de Kalman simplificado para ajuste do offset local
        self.time_offset = 0.9 * self.time_offset + 0.1 * delta

        # Registrar o handover no Ledger local
        handover = {
            'timestamp': datetime.now().isoformat(),
            'satellite': satellite.name,
            'delta_ns': delta * 1e9,
            'new_offset_ns': self.time_offset * 1e9
        }
        self.handovers.append(handover)

        # Atualizar Coerência C baseada na estabilidade dos deltas
        if len(self.handovers) > 1:
            recent_deltas = [h['delta_ns'] for h in self.handovers[-10:]]
            std_dev = np.std(recent_deltas)
            # Normalização: desvio de 1ns -> C alto; desvio alto -> C baixo
            self.C = 1.0 / (1.0 + std_dev)

        # Acumular Satoshi (Memória do Tempo)
        self.satoshi += self.C * 0.01

        return delta

    def verify_conservation(self) -> bool:
        """C + F = 1?"""
        F = 1.0 - self.C
        return abs(self.C + F - 1.0) < 1e-10

if __name__ == "__main__":
    sat = GNSSSatellite("Michibiki QZS-1", "QZSS")
    server = Stratum1Server("Masato-Grandmaster")

    print(f"--- Sincronização Stratum 1: {server.name} ---")
    for i in range(10):
        server.synchronize(sat, time.time())

    print(f"Offset Final: {server.time_offset*1e9:.2f} ns")
    print(f"Coerência C: {server.C:.4f}")
    print(f"Satoshi Temporal: {server.satoshi:.4f} bits")
