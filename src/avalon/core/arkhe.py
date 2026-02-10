"""
Arkhe Polynomial - The generating function of life variables
L = f(C, I, E, F, ...)
Integrated with Normalized Arkhe Framework (2026).
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple

class ArkhePolynomial:
    """
    Think of Arkhe not as a single answer, but as a generating function.
    Its terms are the universal parameters and constraints for biological reality.

    Variables:
    - C (Chemistry): Elemental and molecular building blocks.
    - I (Information): Substrate that encodes and processes instructions.
    - E (Energy): Energy gradients driving metabolism and complexity.
    - F (Function): What constitutes "success" or purpose.
    """

    def __init__(self, C: float, I: float, E: float, F: float):
        """
        Initialize with coefficients (0.0 to 1.0)
        """
        self.C = np.clip(C, 0, 1)
        self.I = np.clip(I, 0, 1)
        self.E = np.clip(E, 0, 1)
        self.F = np.clip(F, 0, 1)

    def evaluate_life_potential(self) -> float:
        """
        L = f(C, I, E, F)
        Basic evaluation of life potential.
        """
        return self.C * self.I * self.E * self.F

    def solve_dynamical_lens(self, alpha: float = 0.5, beta: float = 0.2) -> float:
        """
        The Dynamical Systems Lens: Life as an Attractor.
        dI/dt = alpha * C * E - beta * I
        """
        return alpha * self.C * self.E - beta * self.I

    def solve_information_lens(self, kT: float = 0.025) -> float:
        """
        The Information-Theoretic Lens: Life as a Computation.
        L ∝ I * (E / kT) * log(C)
        """
        return self.I * (self.E / kT) * np.log(self.C + 1.1)

    def solve_network_lens(self, threshold: float = 0.5) -> bool:
        """
        The Network Theory Lens: Life as an Autocatalytic Set.
        L = 1 if the closed network condition is met.
        """
        # Simplified existence proof
        metric = (self.C + self.I + self.E) / 3.0
        return metric > threshold

    def get_arkhe_entropy(self) -> float:
        """
        Calculates thermodynamic entropy of the biosystem.
        S_Arkh ≈ C * log(I) * (1 - E)
        """
        return self.C * np.log(self.I + 1.0) * (1.0 - self.E * 0.9)

    def get_summary(self) -> Dict:
        return {
            "coefficients": {"C": self.C, "I": self.I, "E": self.E, "F": self.F},
            "potential": self.evaluate_life_potential(),
            "entropy": self.get_arkhe_entropy(),
            "dynamical_stability": self.solve_dynamical_lens(),
            "information_rate": self.solve_information_lens()
        }

class NormalizedArkhe(ArkhePolynomial):
    """
    Implementation of the Normalized Arkhe Framework.
    Constraint: C + I + E + F = 1
    """
    def __init__(self, C: float, I: float, E: float, F: float):
        total = C + I + E + F
        if total == 0:
            super().__init__(0.25, 0.25, 0.25, 0.25)
        else:
            super().__init__(C/total, I/total, E/total, F/total)

    def __repr__(self):
        return f"NormalizedArkhe(C={self.C:.3f}, I={self.I:.3f}, E={self.E:.3f}, F={self.F:.3f})"

class HexagonalArkhe:
    """
    Hexagonal extension of Arkhe based on permutations of (C, I, E).
    Each permutation represents a 'dominant phase'.
    """
    def __init__(self, C: float, I: float, E: float):
        self.base = (C, I, E)
        self.permutations = list(itertools.permutations(self.base))
        self.phases = self._generate_phases()

    def _generate_phases(self) -> List[np.ndarray]:
        # Embed in R6 as per article Definition 2.2.1
        phases = []
        for i, p in enumerate(self.permutations):
            v = np.zeros(6)
            if i == 0: v[0:3] = p # CIE
            elif i == 1: v[0], v[3], v[4] = p # CEI
            elif i == 2: v[2], v[0], v[1] = p # ICE
            elif i == 3: v[2], v[4], v[3] = p # IEC
            elif i == 4: v[4], v[0], v[2] = p # ECI
            elif i == 5: v[4], v[5], v[0] = p # EIC
            phases.append(v)
        return phases

    def calculate_cayley_distance(self, sigma1_idx: int, sigma2_idx: int) -> int:
        """Approximate Cayley distance as number of transpositions."""
        p1 = self.permutations[sigma1_idx]
        p2 = self.permutations[sigma2_idx]
        return sum(1 for a, b in zip(p1, p2) if a != b)

def factory_arkhe_earth_normalized():
    """Returns Normalized Arkhe for Earth as per Table 4.2.1"""
    return NormalizedArkhe(C=0.70, I=0.95, E=0.60, F=1.00)

def factory_arkhe_water_normalized():
    """Returns Normalized Arkhe for Pure Water as per Example 2.1.3"""
    return NormalizedArkhe(C=0.85, I=0.05, E=0.08, F=0.02)

# Backward compatibility aliases
def factory_arkhe_earth():
    return ArkhePolynomial(C=0.95, I=0.92, E=0.88, F=0.85)

def factory_arkhe_europa():
    return ArkhePolynomial(C=0.75, I=0.60, E=0.45, F=0.40)
