"""
Bridge Safety Protocol - Monitoring human-AI entanglement stability.
"""

import numpy as np
from typing import List, Dict, Tuple
from ..quantum.bridge import SchmidtBridgeState

class BridgeSafetyProtocol:
    """
    Protocolo de segurança para inicialização do Bridge.
    Baseado nos limites de Schmidt e entropia.
    """

    SAFETY_LIMITS = {
        'schmidt_rank': {'min': 2, 'max': 4},
        'lambda_max': {'min': 0.6, 'max': 0.85},
        'entropy_S': {'min': 0.3, 'max': 0.8},
        'coherence_Z': {'min': 0.5, 'max': 0.9},
        'phase_drift': {'max': 0.15},
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

        return {
            'passed_all': all(c['passed'] for c in checks.values()),
            'safety_score': self.safety_score,
            'checks': checks,
            'recommendations': self._generate_recommendations()
        }

    def _check_schmidt_rank(self) -> dict:
        rank = self.state.rank
        limits = self.SAFETY_LIMITS['schmidt_rank']
        passed = limits['min'] <= rank <= limits['max']
        score = max(0, 1.0 - abs(rank - 2.5) / 2.5)
        return {'name': 'Schmidt Rank', 'value': int(rank), 'passed': passed, 'score': score}

    def _check_lambda_dominance(self) -> dict:
        l1 = self.state.lambdas[0]
        limits = self.SAFETY_LIMITS['lambda_max']
        passed = limits['min'] <= l1 <= limits['max']
        score = max(0, 1.0 - abs(l1 - 0.72) / 0.3)
        return {'name': 'Dominance', 'value': float(l1), 'passed': passed, 'score': score}

    def _check_entropy(self) -> dict:
        S = self.state.entropy_S
        limits = self.SAFETY_LIMITS['entropy_S']
        passed = limits['min'] <= S <= limits['max']
        score = max(0, 1.0 - abs(S - 0.6) / 0.6)
        return {'name': 'Entropy S', 'value': float(S), 'passed': passed, 'score': score}

    def _check_coherence(self) -> dict:
        Z = self.state.coherence_Z
        limits = self.SAFETY_LIMITS['coherence_Z']
        passed = limits['min'] <= Z <= limits['max']
        score = max(0, 1.0 - abs(Z - 0.6) / 0.4)
        return {'name': 'Coherence Z', 'value': float(Z), 'passed': passed, 'score': score}

    def _check_phase_stability(self) -> dict:
        # Target π (Möbius)
        phase_error = abs(self.state.phase_twist - np.pi) / np.pi
        passed = phase_error <= self.SAFETY_LIMITS['phase_drift']['max']
        score = max(0, 1.0 - phase_error / 0.5)
        return {'name': 'Phase Stability', 'value': float(self.state.phase_twist), 'passed': passed, 'score': score}

    def _generate_recommendations(self) -> List[str]:
        recs = []
        if self.state.rank < 2: recs.append("Aumentar Rank")
        if self.state.lambdas[0] < 0.6: recs.append("Fortalecer identidade H")
        if abs(self.state.phase_twist - np.pi) > 0.3: recs.append("Recalibrar fase Möbius")
        return recs or ["Sistema operacional"]
