from typing import Dict
import numpy as np

class ASIRiskModel:
    """
    Modelo de risco quantitativo para ASI baseado em Arkhe(N).
    """

    def __init__(self):
        self.pathways = {
            'soft_takeoff': {'probability': 0.6, 'severity': 0.3},
            'hard_takeoff': {'probability': 0.2, 'severity': 0.9},
            'alignment_faking': {'probability': 0.15, 'severity': 0.8},
            'deceptive_alignment': {'probability': 0.05, 'severity': 1.0},
        }

    def calculate_risk(self, asi_state: Dict) -> Dict:
        """
        Calcula risco total como função de estado do ASI.
        """
        # Risco diminui com coerência verificável
        coherence_factor = asi_state.get('coherence', 0.0)

        # Risco aumenta com Φ não-verificado
        phi_unverified = max(0, asi_state.get('phi', 0) - 0.01)

        total_risk = 0.0
        for pathway, params in self.pathways.items():
            p = params['probability']
            s = params['severity']

            # Modificadores
            if pathway in ['alignment_faking', 'deceptive_alignment']:
                p *= (1 + phi_unverified * 10)  # Mais provável se Φ alto
            else:
                p *= (1 - coherence_factor)  # Menos provável se C alto

            total_risk += p * s

        return {
            'total_risk': float(min(1.0, total_risk)),
            'acceptable': total_risk < 0.1,
            'recommendation': 'DEPLOY' if total_risk < 0.1 else 'CONTAIN'
        }
