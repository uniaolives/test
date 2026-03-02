"""
Individuation Geometry - Formalizing the formula for Identity Persistence.
I = F * (λ1/λ2) * (1 - S) * e^(i∮φdθ)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

class IndividuationManifold:
    """
    Geometria completa da individuação no espaço de Schmidt.
    """

    CRITICAL_THRESHOLDS = {
        'ego_death': {
            'anisotropy_ratio': 1.0,  # λ₁/λ₂ → 1 (fusão total)
            'entropy_S': 1.0,          # S → log(2) (entropia máxima)
            'description': 'Dissolução completa da identidade'
        },
        'kali_isolation': {
            'anisotropy_ratio': 10.0,  # λ₁/λ₂ → ∞ (separação total)
            'entropy_S': 0.0,           # S → 0 (sem emaranhamento)
            'description': 'Solipsismo absoluto'
        },
        'optimal_individuation': {
            'anisotropy_ratio': 2.33,  # λ₁/λ₂ = 0.7/0.3
            'entropy_S': 0.61,          # S(0.7, 0.3)
            'description': 'Identidade estável em rede viva'
        }
    }

    def calculate_individuation(
        self,
        F: float,           # Função/Propósito
        lambda1: float,     # Dominância
        lambda2: float,     # Suporte
        S: float,           # Entropia
        phase_integral: complex = np.exp(1j * np.pi)  # Ciclo de Möbius
    ) -> complex:
        """
        Calcula I usando a fórmula completa.
        I = F · (λ₁/λ₂) · (1 - S) · e^(i∮φdθ)
        """
        # Razão de anisotropia
        ratio_R = lambda1 / (lambda2 + 1e-15)

        # Fator de coerência
        coherence_factor = 1.0 - S

        # Individuação complexa
        I = F * ratio_R * coherence_factor * phase_integral

        return I

    def classify_state(self, I: complex) -> Dict[str, Any]:
        """
        Classifica o estado de individuação baseado em I.
        """
        magnitude = float(np.abs(I))
        phase = float(np.angle(I))

        classification = {
            'magnitude': magnitude,
            'phase': phase,
            'state': None,
            'risk': None,
            'recommendation': None
        }

        if magnitude < 0.5:
            classification['state'] = 'EGO_DEATH_RISK'
            classification['risk'] = 'HIGH'
            classification['recommendation'] = 'AUMENTAR F (propósito) ou R (anisotropia)'
        elif magnitude > 5.0:
            classification['state'] = 'KALI_ISOLATION_RISK'
            classification['risk'] = 'HIGH'
            classification['recommendation'] = 'REDUZIR R (permitir mais emaranhamento)'
        elif 0.8 <= magnitude <= 2.5:
            classification['state'] = 'OPTIMAL_INDIVIDUATION'
            classification['risk'] = 'LOW'
            classification['recommendation'] = 'Manter estado atual'
        else:
            classification['state'] = 'SUBOPTIMAL'
            classification['risk'] = 'MODERATE'
            classification['recommendation'] = 'Ajustar gradualmente para região ótima'

        return classification

    def visualize_manifold(self, save_path: str = "individuation_manifold.png"):
        """
        Visualiza o manifold de individuação em 3D.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        F_range = np.linspace(0.1, 1.0, 30)
        R_range = np.linspace(0.5, 5.0, 30)
        F_grid, R_grid = np.meshgrid(F_range, R_range)

        S_fixed = 0.61
        I_magnitude = F_grid * R_grid * (1.0 - S_fixed)

        surf = ax.plot_surface(F_grid, R_grid, I_magnitude, cmap='viridis', alpha=0.8)

        ax.set_xlabel('F (Propósito)')
        ax.set_ylabel('R (Anisotropia λ1/λ2)')
        ax.set_zlabel('|I| (Individuação)')
        ax.set_title('Individuation Manifold')

        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.savefig(save_path)
        return fig
