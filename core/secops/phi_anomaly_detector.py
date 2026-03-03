# core/secops/phi_anomaly_detector.py
import numpy as np
from typing import List, Dict, Tuple
from core.arkhe_physics.entropy_unit import ArkheEntropyUnit

class PhiAnomalyDetector:
    """
    Detector de anomalias baseado em violações da eficiência Φ.
    Não apenas detecta padrões estatísticos, mas desvios da lei termodinâmica.
    """

    def __init__(self, omni_kernel=None):
        self.kernel = omni_kernel
        self.phi_history: List[Tuple[float, float]] = []  # (tempo, Φ)

    def analyze_handover_stream(self, handovers: List[Dict]) -> List[Dict]:
        """
        Analisa handovers em busca de violações de Φ = E/T.

        Uma anomalia Arkhe ocorre quando:
        1. Φ < Φ_min (ineficiência extrema - possível ataque de negação)
        2. Φ > Φ_max (eficiência impossível - possível falsificação)
        3. dΦ/dt >> 0 (transição abrupta - possível colusão)
        """
        alerts = []

        for i, h in enumerate(handovers):
            # Calcula Φ deste handover
            energy = h.get('energy_j', 0)
            time_ms = h.get('time_ms', 1)
            phi = energy / time_ms if time_ms > 0 else 0

            # Verifica violações
            phi_min, phi_max = self._get_phi_bounds(h['source_layer'])

            if phi < phi_min:
                alerts.append(self._create_alert(
                    h, 'INEFFICIENCY_ATTACK',
                    f'Φ={phi:.2e} < Φ_min={phi_min:.2e}',
                    severity='CRITICAL'
                ))

            elif phi > phi_max:
                alerts.append(self._create_alert(
                    h, 'IMPOSSIBLE_EFFICIENCY',
                    f'Φ={phi:.2e} > Φ_max={phi_max:.2e}',
                    severity='CRITICAL'
                ))

            # Detecta mudanças abruptas (colusão)
            if i > 0:
                phi_prev = self.phi_history[-1][1]
                d_phi = abs(phi - phi_prev) / phi_prev if phi_prev > 0 else 0

                if d_phi > 10.0:  # Mudança de 1000%
                    alerts.append(self._create_alert(
                        h, 'PHASE_TRANSITION',
                        f'dΦ/dt = {d_phi:.1f}x (possível colusão)',
                        severity='HIGH'
                    ))

            self.phi_history.append((h['timestamp'], phi))

        return alerts

    def _get_phi_bounds(self, layer: str) -> Tuple[float, float]:
        """Limites de eficiência por camada (física do Arkhe Protocol)"""
        bounds = {
            'engineering': (1e-6, 1e-2),   # Atuadores: baixa eficiência tolerável
            'devops': (1e-4, 1e-1),        # Software: eficiência média
            'secops': (1e-3, 0.5),         # Análise: alta eficiência necessária
        }
        return bounds.get(layer, (1e-6, 1.0))

    def _create_alert(self, handover, type, msg, severity):
        return {
            'handover_id': handover.get('id'),
            'type': type,
            'message': msg,
            'severity': severity,
            'timestamp': handover.get('timestamp')
        }
