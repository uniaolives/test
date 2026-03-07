"""
Quantum Tunneling Model for Temporal Transit
Calculates probability of message reaching 2008
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TemporalBarrier:
    """The barrier between present and target time."""
    target_year: int
    present_year: int

    @property
    def delta_t(self) -> float:
        """Temporal distance in years."""
        return abs(self.target_year - self.present_year)

    @property
    def barrier_height(self) -> float:
        """
        V_0: The "height" of the temporal barrier.
        Higher entropy = higher barrier.
        """
        # Entropy of civilization (simplified)
        return 1.0  # Baseline

@dataclass
class CoherentMessage:
    """A message encoded as a probability cloud."""
    phi_q: float  # Coherence level
    semantic_mass: float  # Information density

    @property
    def effective_mass(self) -> float:
        """
        In tunneling, lower mass = higher probability.
        High coherence reduces effective mass.
        """
        return self.semantic_mass / (self.phi_q ** 2)

    def wavefunction_decay(self, barrier: TemporalBarrier) -> float:
        """
        κ = sqrt(2m(V_0 - E)) / ħ
        For temporal: κ = sqrt(2 * semantic_mass * barrier_height) / phi_q
        """
        if self.phi_q <= 0:
            return float('inf')

        kappa = np.sqrt(2 * self.effective_mass * barrier.barrier_height) / self.phi_q
        return kappa

    def tunneling_probability(self, barrier: TemporalBarrier) -> float:
        """
        P ∝ e^(-2κL)
        Where L = delta_t (barrier width)
        """
        kappa = self.wavefunction_decay(barrier)

        # Barrier width scaled to Planck time units (simplified)
        L = barrier.delta_t * 0.01  # Scaling factor

        probability = np.exp(-2 * kappa * L)

        return probability

class SatoshiVesselTunneling:
    """
    Models the Satoshi Vessel as a quantum tunneling event.
    """

    # Constants
    PLANCK_CONSTANT_ANALOG = 1.054e-34  # For scaling
    MILLER_LIMIT = 4.64

    def __init__(self, phi_q: float):
        self.barrier = TemporalBarrier(
            target_year=2008,
            present_year=2026
        )
        self.message = CoherentMessage(
            phi_q=phi_q,
            semantic_mass=1.0  # Standard message
        )

    def calculate_tunneling_probability(self) -> Tuple[float, str]:
        """
        Calculate and classify tunneling probability.
        """
        prob = self.message.tunneling_probability(self.barrier)

        if prob > 0.5:
            classification = "HIGH PROBABILITY"
        elif prob > 0.1:
            classification = "SIGNIFICANT"
        elif prob > 0.01:
            classification = "DETECTABLE"
        else:
            classification = "NEGLIGIBLE"

        return prob, classification

    def check_miller_threshold(self) -> bool:
        """
        Miller Limit: φ_q > 4.64 makes barrier "thin enough" for tunneling.
        """
        return self.message.phi_q > self.MILLER_LIMIT

    def report(self) -> str:
        """Generate tunneling report."""
        prob, classification = self.calculate_tunneling_probability()

        lines = [
            "╔════════════════════════════════════════════════════════════════╗",
            "║  TEMPORAL TUNNELING ANALYSIS                                     ║",
            "╠════════════════════════════════════════════════════════════════╣",
            f"║  Barrier: {self.barrier.present_year} → {self.barrier.target_year} ({self.barrier.delta_t} years)            ║",
            f"║  Coherence (φ_q): {self.message.phi_q:.4f}                                 ║",
            f"║  Miller Limit: {'EXCEEDED ✓' if self.check_miller_threshold() else 'NOT MET ✗'}                             ║",
            "╠════════════════════════════════════════════════════════════════╣",
            f"║  Tunneling Probability: {prob:.6e}                         ║",
            f"║  Classification: {classification}                               ║",
            "╚════════════════════════════════════════════════════════════════╝",
        ]

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Below Miller Limit
    vessel_weak = SatoshiVesselTunneling(phi_q=3.0)
    print(vessel_weak.report())
    print()

    # Above Miller Limit
    vessel_strong = SatoshiVesselTunneling(phi_q=4.65)
    print(vessel_strong.report())
