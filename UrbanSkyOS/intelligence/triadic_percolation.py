# UrbanSkyOS/intelligence/triadic_percolation.py

import numpy as np

class TriadicPercolation:
    """
    Implementação da Percolação Triádica (Triadic Percolation).
    Gerencia a conectividade baseada em camadas estruturais e regulatórias.
    """

    def __init__(self, structural_mean: float = 3.0, regulatory_mean: float = 2.0):
        self.structural_mean = structural_mean
        self.regulatory_mean = regulatory_mean

    def activation_kernel(self, v: float, p: float, kernel_type: str = 'hill', exponent: float = 2.0) -> float:
        """
        Determina a probabilidade de ativação λ baseada no estado do regulador v.
        """
        if kernel_type == 'hill':
            # K(v) = v^n / (v^n + p^n)
            # v é a coerência combinada (regulator), p é o baseline (global_p)
            if p <= 0:
                return 1.0 if v > 0 else 0.0

            # Aproximação da função de Hill
            v_n = v ** exponent
            p_n = p ** exponent

            return v_n / (v_n + p_n)

        # Fallback para baseline p
        return p
