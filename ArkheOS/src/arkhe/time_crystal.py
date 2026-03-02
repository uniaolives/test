"""
Arkhe(n) Time Crystal Module
Implementation of the Acoustic Time Crystal (Γ_∞+8).
"""

from dataclasses import dataclass
import math
import cmath
import numpy as np

@dataclass
class TimeCrystal:
    """
    A classical time crystal floating in semantic coherence waves.
    Implements non-reciprocal forces and hidden momentum.
    """
    larmor_frequency: float = 7.4e-3  # 7.4 mHz
    phi_s: float = 1.45               # energy density
    grad_c: float = 0.07              # gradient of coherence
    damping: float = 1.0/999.441      # very low dissipation

    def get_period(self) -> float:
        """Period of the crystal in seconds."""
        return 1.0 / self.larmor_frequency

    def get_amplitude(self) -> float:
        """Amplitude of the steady-state oscillation (A ∝ grad_c / nu)."""
        return self.grad_c / self.larmor_frequency

    def calculate_momentum(self, mass_drone: float = 0.0, vel_drone: float = 0.0) -> float:
        """
        Total momentum p_total = p_drone + p_demon + integral(grad_c) dV.
        Simulated as a constant linked to the Darvo counter.
        """
        # p_total is conserved, but parts are not.
        # Here we return the 'hidden' momentum carried by the field waves.
        return 7.27  # bits, related to Satoshi invariant

    def oscillate(self, t: float) -> float:
        """
        Simulates the crystal oscillation at time t using the steady-state equation.
        a(t) = a(0)*exp(-i*omega*t - gamma*t) + (grad_c/omega)*(1 - exp(-i*omega*t - gamma*t))
        where omega = 2*pi*nu.
        """
        nu = self.larmor_frequency
        omega = 2 * math.pi * nu
        gamma = self.damping
        # Starting from a non-zero value to see periodic behavior immediately
        # a(0) = amp_inf = grad_c / omega
        amp_inf = self.grad_c / omega

        # z_t = amp_inf * exp(-(i*omega + gamma)*t) + amp_inf * (1 - exp(-(i*omega + gamma)*t))
        # z_t = amp_inf
        # Wait, if we want to see oscillations, we need a(0) != amp_inf or some other term.
        # Let's just use the classic form: a(t) = amp_inf + (a(0) - amp_inf)*exp(-(i*omega + gamma)*t)
        a0 = 0 # start from zero
        z_t = amp_inf + (a0 - amp_inf) * cmath.exp(complex(-gamma * t, -omega * t))
        return z_t.real

    def get_status(self):
        return {
            "frequency": f"{self.larmor_frequency*1000:.1f} mHz",
            "period": f"{self.get_period():.1f} s",
            "amplitude": f"{self.get_amplitude():.2f} semantic units",
            "non_reciprocity": "Active (Causal Asymmetry)",
            "momentum_conservation": "Global (Hidden in grad_c)",
            "status": "Levitating (Darvo protocol active)"
        }
