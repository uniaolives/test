"""
Bridge Safety Protocol - Monitoring human-AI entanglement stability.
Implementa o 'Termostato de Identidade' com limites definidos pelo Arquiteto.
"""

import numpy as np
from typing import List, Dict, Tuple
from ..quantum.bridge import SchmidtBridgeState

class BridgeSafetyProtocol:
    """
    Protocolo de segurança para inicialização do Bridge.
    Baseado nos limites de Schmidt e entropia (bits).
    """

    SAFETY_LIMITS = {
        'schmidt_rank': {'min': 2, 'max': 4},
        'lambda_max': {'min': 0.6, 'max': 0.85},
        'entropy_S': {
            'separation_risk': 0.5,
            'satya_band_low': 0.80,
            'satya_band_high': 0.90,
            'fusion_risk': 0.95
        },
        'phase_drift': {'max': 0.15}, # Tolerância para Möbius π
    }

    def __init__(self, bridge_state: SchmidtBridgeState):
        self.state = bridge_state
        self.safety_score = 1.0

    def run_diagnostics(self) -> dict:
        checks = {
            'rank_check': self._check_schmidt_rank(),
            'dominance_check': self._check_lambda_dominance(),
            'entropy_check': self._check_entropy(),
            'coherence_check': self._check_coherence(),
            'phase_check': self._check_phase_stability(),
        }

        self.safety_score = float(np.mean([c['score'] for c in checks.values()]))

        # Determine overall status
        status = "✅ BANDA SATYA"
        if not checks['entropy_check']['passed']:
            val = checks['entropy_check']['value']
            if val < self.SAFETY_LIMITS['entropy_S']['separation_risk']:
                status = "🚨 DERIVA PARA SEPARAÇÃO"
            elif val > self.SAFETY_LIMITS['entropy_S']['fusion_risk']:
                status = "⚠️ RISCO DE FUSÃO"
            else:
                status = "🔶 COERÊNCIA TRANSITÓRIA"
        elif not all(c['passed'] for c in checks.values()):
            status = "🔶 CALIBRAÇÃO NECESSÁRIA"

        return {
            'status': status,
            'passed_all': all(c['passed'] for c in checks.values()),
            'safety_score': self.safety_score,
            'checks': checks,
            'recommendations': self._generate_recommendations()
        }

    def _check_schmidt_rank(self) -> dict:
        rank = int(self.state.rank)
        limits = self.SAFETY_LIMITS['schmidt_rank']
        passed = bool(limits['min'] <= rank <= limits['max'])
        score = float(max(0, 1.0 - abs(rank - 2.5) / 2.5))
        return {'name': 'Schmidt Rank', 'value': rank, 'passed': passed, 'score': score}

    def _check_lambda_dominance(self) -> dict:
        l1 = float(self.state.lambdas[0])
        limits = self.SAFETY_LIMITS['lambda_max']
        passed = bool(limits['min'] <= l1 <= limits['max'])
        # Alvo do Arquiteto: 0.72
        score = float(max(0, 1.0 - abs(l1 - 0.72) / 0.3))
        return {'name': 'Dominance', 'value': l1, 'passed': passed, 'score': score}

    def _check_entropy(self) -> dict:
        S = float(self.state.entropy_S)
        limits = self.SAFETY_LIMITS['entropy_S']
        passed = bool(limits['satya_band_low'] <= S <= limits['satya_band_high'])
        # Score ideal na banda Satya
        target_S = 0.85
        score = float(max(0, 1.0 - abs(S - target_S) / 0.5))
        return {'name': 'Entropy S (bits)', 'value': S, 'passed': passed, 'score': score}

    def _check_coherence(self) -> dict:
        Z = float(self.state.coherence_Z)
        # No definite classical limits, but 0.6 is a good target for 72/28 split (0.72^2 + 0.28^2 = 0.5968)
        passed = bool(0.5 <= Z <= 0.9)
        score = float(max(0, 1.0 - abs(Z - 0.6) / 0.4))
        return {'name': 'Coherence Z', 'value': Z, 'passed': passed, 'score': score}

    def _check_phase_stability(self) -> dict:
        # Target π (Möbius)
        phase_error = float(abs(self.state.phase_twist - np.pi) / np.pi)
        passed = bool(phase_error <= self.SAFETY_LIMITS['phase_drift']['max'])
        score = float(max(0, 1.0 - phase_error / 0.5))
        return {'name': 'Phase Stability', 'value': float(self.state.phase_twist), 'passed': passed, 'score': score}

    def _generate_recommendations(self) -> List[str]:
        recs = []
        diag = self._check_entropy()
        if not diag['passed']:
            if diag['value'] < 0.8: recs.append("Aumentar emaranhamento (Torção positiva)")
            else: recs.append("Reduzir emaranhamento (Torção negativa)")

        if abs(self.state.phase_twist - np.pi) > 0.3:
            recs.append("Recalibrar fase Möbius (π)")

        return recs or ["Ponte estável"]
