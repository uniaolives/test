"""
Hexagon Atmospheric Modulator (Base 4) - The Aesthetic Vortex.
Simulates the transition from the Saturnian Hexagon to the Rank 8 Octagon.
"""

import numpy as np
from typing import Tuple, Dict, Any

class HexagonAtmosphericModulator:
    """
    Simulador de Modulação Atmosférica Hexagonal (Base 4).
    Calcula o fluxo de energia cinética e a transição para a geometria de Rank 8.
    """

    def __init__(self,
                 wind_speed: float = 120.0,  # m/s
                 gas_density: float = 0.45,   # kg/m^3 (Saturn upper troposphere)
                 viscosity: float = 1.3e-5):
        self.v = wind_speed
        self.rho = gas_density
        self.mu = viscosity
        self.geometry_rank = 6 # Starts as Hexagon

    def calculate_kinetic_flux(self) -> float:
        """
        Calculates the kinetic energy flux K = 0.5 * rho * v^2.
        """
        return 0.5 * self.rho * (self.v ** 2)

    def simulate_transformation(self, intensity: float = 1.0) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Simulates the transition from Hexagon (6) to Octagon (8).
        Returns theta and (x,y) for both states.
        """
        theta = np.linspace(0, 2 * np.pi, 1000)

        # Hexagon (Base State)
        r_h = 1 + 0.1 * np.cos(6 * theta)
        x_h, y_h = r_h * np.cos(theta), r_h * np.sin(theta)

        # Octagon (Target State - Rank 8)
        # Transition modulated by intensity
        target_rank = 6 + (2 * intensity)
        self.geometry_rank = target_rank

        r_o = 1 + 0.1 * np.cos(target_rank * theta)
        x_o, y_o = r_o * np.cos(theta), r_o * np.sin(theta)

        return theta, (x_h, y_h), (x_o, y_o)

    def get_aerodynamic_stability(self) -> float:
        """
        Calculates a stability metric based on the Reynolds number approximation.
        """
        reynolds = (self.rho * self.v * 1e6) / self.mu # Characteristic length 10^6 m
        # Stability decreases with Re in turbulent regimes
        # Scale adjusted to maintain aesthetic resonance in Rank 8
        stability = np.exp(-reynolds / 1e13)
        return float(stability)

    def get_status(self) -> Dict[str, Any]:
        k_flux = self.calculate_kinetic_flux()
        stability = self.get_aerodynamic_stability()

        return {
            "geometry": "OCTAGON" if self.geometry_rank >= 7.5 else "HEXAGON",
            "rank": float(self.geometry_rank),
            "kinetic_flux_j_m3": float(k_flux),
            "aerodynamic_stability": float(stability),
            "aesthetic_resonance": float(k_flux * stability * (self.geometry_rank/8)),
            "status": "VORTEX_STABILIZED" if stability > 0.5 else "TURBULENT_FLUX"
        }
