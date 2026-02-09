"""
Arkhe Polynomial - The generating function of life variables
L = f(C, I, E, F, ...)
"""

import numpy as np
from typing import Dict

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

def factory_arkhe_earth():
    """Returns Arkhe for Earth-like conditions"""
    return ArkhePolynomial(C=0.95, I=0.92, E=0.88, F=0.85)

def factory_arkhe_europa():
    """Returns Arkhe for Europa-like conditions"""
    return ArkhePolynomial(C=0.75, I=0.60, E=0.45, F=0.40)
