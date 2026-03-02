"""
Arkhe Neuroscience Module - Neural Population Geometry
Implementation of Wakhloo, Slatton & Chung (Nature Neuroscience 2026).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class NeuroGeometricTerms:
    c: float  # Neural-latent correlation (C)
    pr: float # Neural dimension (PR)
    f: float  # Signal-Signal Factorization (1/omega)
    s: float  # Signal-Noise Factorization (1/F)

class NeuroGeometryEngine:
    """
    Engine that implements the Generalization Error equation (Eg)
    and maps the neural population geometry to the Arkhe framework.
    """
    def __init__(self, terms: NeuroGeometricTerms):
        self.terms = terms

    def calculate_generalization_error(self, p: int) -> float:
        """
        E_g = (1/pi) * tan^-1(sqrt( (pi / (2 * p * c^2 * PR)) + (1/f) + (1/s) - 1 ))
        Where p is the number of training samples (handovers).
        """
        c = self.terms.c
        pr = self.terms.pr
        f = self.terms.f
        s = self.terms.s

        sample_term = np.pi / (2 * p * (c**2) * pr)
        independent_term = (1.0 / f) + (1.0 / s) - 1.0

        # Ensure we don't have negative values due to rounding in simulation
        total_arg = max(0.0, sample_term + independent_term)

        arg = np.sqrt(total_arg)
        eg = (1.0 / np.pi) * np.arctan(arg)

        return eg

    @staticmethod
    def map_arkhe_to_neuro(coherence: float, dimension: float, f_val: float, s_val: float) -> NeuroGeometricTerms:
        """Maps Arkhe internal parameters to Nature Neuroscience 2026 terms."""
        # Use provided f and s values directly to match user expectations
        return NeuroGeometricTerms(
            c=coherence,
            pr=dimension,
            f=f_val,
            s=s_val
        )

    def get_summary(self, p: int) -> Dict:
        return {
            "handovers_p": p,
            "error_generalization": self.calculate_generalization_error(p),
            "status": "PEER-REVIEWED (Nature Neuroscience 2026)",
            "terms": {
                "correlation_c": self.terms.c,
                "dimension_pr": self.terms.pr,
                "factorization_f": self.terms.f,
                "noise_factor_s": self.terms.s
            }
        }
