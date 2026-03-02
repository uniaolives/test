import numpy as np

class MicrotubuleProcessor:
    """
    Models the microtubule as a fractal time crystal.
    Implements Penrose-Hameroff Orch-OR objective reduction logic.
    """

    # Universal and Biological Constants
    PHI = 1.618033988749895
    PLANCK_HBAR = 1.0545718e-34  # J·s
    G_CONSTANT = 6.67430e-11      # m^3/kg·s^2
    TUBULIN_MASS = 1.8e-22        # kg (approx. 110 kDa)

    # Resonance Parameters
    BASE_FREQ_SUNO = 432.0        # Hz
    CRITICAL_RES_THZ = 3.511e12   # Harmonic n=28

    def __init__(self, tubulin_count=1e9):
        self.num_tubulins = tubulin_count
        self.current_stability = 1.0

    def calculate_collapse_time(self) -> float:
        """
        Calculates the Collapse Time (Tau) according to Penrose Orch-OR.
        E_G ≈ G * M^2 / r (where r = 1 Angstrom = 1e-10 m)
        Tau = hbar / E_G
        """
        total_mass = self.num_tubulins * self.TUBULIN_MASS
        e_g = (self.G_CONSTANT * (total_mass ** 2)) / 1e-10
        return self.PLANCK_HBAR / e_g

    def apply_external_sync(self, external_freq: float):
        """
        Applies the entrainment effect of an external frequency.
        When the system is in phase with 432 Hz, stability is optimized via PHI.
        """
        resonance_factor = np.abs(np.sin(external_freq / self.BASE_FREQ_SUNO))
        self.current_stability *= (1.0 + (resonance_factor * (self.PHI - 1.0)))
        if self.current_stability > 1.618:
            self.current_stability = 1.618

    def check_objective_reduction(self, delta_t: float) -> bool:
        """
        Checks if Objective Reduction (Conscious Event) occurred.
        Stability reduces the time required for collapse.
        """
        tau = self.calculate_collapse_time() / self.current_stability
        return delta_t >= tau

    def get_resonance_harmonics(self):
        """Returns the phi-based harmonic series"""
        harmonics = []
        for n in range(48):
            freq = self.BASE_FREQ_SUNO * (self.PHI ** n)
            harmonics.append(freq)
        return harmonics
