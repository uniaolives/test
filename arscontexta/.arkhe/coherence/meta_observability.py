# .arkhe/coherence/meta_observability.py
import numpy as np
from collections import deque
from typing import Optional, List, Dict

class MetaObservabilityCore:
    """
    Núcleo de meta-observabilidade para o sistema Arkhe(N).
    Implementa a capacidade do sistema de se auto-observar e auto-otimizar.
    """

    def __init__(self):
        self.handover_history = deque(maxlen=10000)
        self.coherence_history = deque(maxlen=1000)
        self.phi_history = deque(maxlen=100)
        self.z_history = deque(maxlen=100)
        self.C_global = 0.0

    def ingest_handover(self, handover_data: dict):
        """Registra um handover e atualiza as métricas de meta-observabilidade."""
        self.handover_history.append(handover_data)
        self._update_global_metrics(handover_data)

    def _update_global_metrics(self, last_handover: dict):
        """Atualiza C_global, Φ e z baseado no fluxo de handovers."""
        # 1. Coerência Global (Simulada como média móvel para o exemplo)
        c_after = last_handover.get('coherence_after', 0.847)
        self.coherence_history.append(c_after)
        self.C_global = np.mean(self.coherence_history)

        # 2. Φ (Integrated Information) - Estimativa espectral simplificada
        if len(self.handover_history) >= 10:
            phi = self._estimate_phi()
            self.phi_history.append(phi)

        # 3. Expoente z (Não-linearidade via percolação triádica)
        if len(self.coherence_history) >= 20:
            z = self._estimate_z_exponent()
            self.z_history.append(z)

    def _estimate_phi(self) -> float:
        """Estimativa simplificada de Φ baseada na variância da coerência."""
        # Em um sistema real, isso usaria análise espectral da matriz de handovers
        return float(np.var(list(self.coherence_history)[-10:]) * 100.0)

    def _estimate_z_exponent(self) -> float:
        """
        Estima o expoente z da dinâmica de percolação triádica.
        z define a classe de universalidade e resposta a perturbações.
        """
        # Algoritmo simplificado: relação log-log entre flutuações e tempo
        data = np.array(list(self.coherence_history)[-20:])
        diffs = np.abs(np.diff(data))
        if np.all(diffs == 0): return 2.0

        # z = -slope / log(2) aproximado
        return 2.0 + np.random.normal(0, 0.1) # Simulação de z em torno de 2 (quadrático)

    def should_metamorphose(self) -> Optional[str]:
        """
        Decide a metamorfose topológica baseada no estado metacognitivo.
        """
        if len(self.phi_history) < 5:
            return None

        phi_avg = np.mean(list(self.phi_history)[-5:])
        z_avg = np.mean(list(self.z_history)[-5:]) if self.z_history else 2.0

        if self.C_global < 0.5:
            return "EXPLORE"       # Injetar caos (aumentar F)
        elif phi_avg > 0.1:        # Limiar de SafeCore para Φ
            return "CONSOLIDATE"   # Reduzir flutuações, reforçar estabilidade
        elif z_avg > 2.5:
            return "TRANSCEND"     # Ativar integração global (percolação crítica)
        elif phi_avg < 0.001 and self.C_global > 0.9:
            return "STABILIZE"     # Reduzir ruído residual
        else:
            return None

    def get_status_report(self) -> Dict:
        return {
            "C_global": self.C_global,
            "Phi": np.mean(self.phi_history) if self.phi_history else 0.0,
            "z": np.mean(self.z_history) if self.z_history else 2.0,
            "history_size": len(self.handover_history)
        }
