import numpy as np
from core.python.arkhe_physics.entropy_unit import ArkheEntropyUnit

class PhiAnomalyDetector:
    def __init__(self, kernel=None):
        self.kernel = kernel
        self.phi_history = []  # (timestamp, phi)

    def analyze_handover(self, handover: dict) -> dict | None:
        """
        Retorna alerta se Φ violar os limites termodinâmicos esperados.
        """
        energy = handover.get('energy_j', 0)
        time_ms = handover.get('duration_ms', 1)
        phi = energy / time_ms if time_ms > 0 else 0

        # Limites por camada (derivados experimentalmente)
        bounds = {
            'engineering': (1e-6, 1e-2),   # Atuadores físicos
            'devops': (1e-4, 1e-1),        # Orquestração
            'secops': (1e-3, 0.5),         # Análise
        }
        low, high = bounds.get(handover.get('source_layer', 'unknown'), (1e-6, 1.0))

        if phi < low:
            return self._alert(handover, 'INEFICIÊNCIA CRÍTICA', f'Φ={phi:.2e} < {low:.2e}', 'CRITICAL')
        if phi > high:
            return self._alert(handover, 'EFICIÊNCIA IMPOSSÍVEL', f'Φ={phi:.2e} > {high:.2e}', 'CRITICAL')

        # Detecta transições abruptas (possível colusão)
        if self.phi_history:
            last_phi = self.phi_history[-1][1]
            ratio = abs(phi - last_phi) / (last_phi + 1e-10)
            if ratio > 10.0:  # Mudança de 1000%
                return self._alert(handover, 'TRANSIÇÃO DE FASE', f'dΦ/dt = {ratio:.1f}x', 'HIGH')

        self.phi_history.append((handover.get('timestamp', 0), phi))
        return None

    def _alert(self, handover, reason, details, severity):
        return {
            'handover_id': handover.get('id', 'unknown'),
            'reason': reason,
            'details': details,
            'phi': handover.get('energy_j', 0) / handover.get('duration_ms', 1),
            'severity': severity
        }
