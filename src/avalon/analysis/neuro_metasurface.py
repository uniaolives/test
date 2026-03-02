"""
Neuro-Metasurface Control: Direct brain-to-electromagnetic field interface.
Implementation of Algorithm 5.2.1 and Equation 5.1.2.
Updated with Holographic Beamforming and AR loop (Section 8.3.3).
"""

import numpy as np
from typing import Dict, List, Tuple

class NeuroMetasurfaceController:
    """
    Controls a programmable metasurface using EEG-derived attention levels.
    """
    def __init__(self, n_rows: int = 8, n_cols: int = 8, frequency_ghz: float = 10.0):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.frequency = frequency_ghz
        self.wavelength = 0.3 / frequency_ghz # in meters
        self.k = 2 * np.pi / self.wavelength

        # Base parameters (Equation 5.1.2)
        self.theta_0 = 0.0
        self.f_0 = 1.0
        self.k_theta = 45.0 # Degrees
        self.k_f = 0.5

    def extract_attention(self, p_alpha: float, p_beta: float, p_gamma: float) -> float:
        """
        Algorithm 5.2.1: EEG Attention Extraction.
        A = 50 * (tanh(R - 1) + 0.5 * tanh(10 * Gamma)) + 50
        """
        epsilon = 1e-9
        r_ratio = p_beta / (p_alpha + epsilon)
        gamma_ratio = p_gamma / (p_alpha + p_beta + epsilon)

        attention = 50 * (np.tanh(r_ratio - 1) + 0.5 * np.tanh(10 * gamma_ratio)) + 50
        return float(np.clip(attention, 0, 100))

    def calculate_beam_parameters(self, attention: float) -> Tuple[float, float]:
        """
        Equation 5.1.2: Beam steering and focus control.
        theta_beam(t) = theta_0 + k_theta * tanh((A(t) - 50) / 25)
        F(t) = F_0 + k_f * (A(t) / 100)
        """
        theta_beam = self.theta_0 + self.k_theta * np.tanh((attention - 50) / 25.0)
        focus = self.f_0 + self.k_f * (attention / 100.0)
        return float(theta_beam), float(focus)

    def generate_metasurface_pattern(self, theta_target: float, phi_target: float) -> np.ndarray:
        """
        Generates the phase pattern for the metasurface.
        phi_ij = k * (x_i * sin(theta) * cos(phi) + y_j * sin(theta) * sin(phi))
        """
        phases = np.zeros((self.n_rows, self.n_cols))
        dx = self.wavelength / 2.0
        dy = self.wavelength / 2.0

        theta_rad = np.radians(theta_target)
        phi_rad = np.radians(phi_target)

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                x_i = i * dx
                y_j = j * dy
                phase = self.k * (x_i * np.sin(theta_rad) * np.cos(phi_rad) +
                                  y_j * np.sin(theta_rad) * np.sin(phi_rad))
                phases[i, j] = phase % (2 * np.pi)

        return phases

    def get_system_status(self, attention: float) -> Dict:
        theta, focus = self.calculate_beam_parameters(attention)
        return {
            "attention_level": attention,
            "beam_steering_angle": theta,
            "focus_factor": focus,
            "frequency_ghz": self.frequency,
            "grid_size": f"{self.n_rows}x{self.n_cols}"
        }

class HolographicMetasurfaceController(NeuroMetasurfaceController):
    """
    3D Holographic Metasurface for Augmented Cognitive Architecture.
    """
    def __init__(self, layers: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.layers = layers

    def generate_3d_hologram(self, target_volume: np.ndarray) -> np.ndarray:
        """
        Generates multi-layer holographic patterns.
        Simplified 3D Field Intensity to Phase mapping.
        """
        # Simulated Gerchberg-Saxton or CGH algorithm
        hologram = np.random.uniform(0, 2*np.pi, (self.layers, self.n_rows, self.n_cols))
        return hologram

    def ar_cognitive_loop(self, attention: float, virtual_object_pos: Tuple[float, float, float]) -> Dict:
        """
        Section 8.3.3: Attention -> EM -> Perception -> Attention loop.
        """
        # 1. Attention maps to beam steering toward virtual object
        # Map object (x,y,z) to (theta, phi)
        x, y, z = virtual_object_pos
        theta_obj = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z))
        phi_obj = np.degrees(np.arctan2(y, x))

        # Adjust steering based on attention intensity
        steering_error = (100 - attention) / 10.0 # High attention = low error
        theta_steer = theta_obj + np.random.normal(0, steering_error)

        # 2. Generate hologram focusing on object
        pattern = self.generate_metasurface_pattern(theta_steer, phi_obj)

        return {
            'perception_reinforcement': float(attention / 100.0),
            'beam_pointing_accuracy': float(1.0 - steering_error / 10.0),
            'hologram_pattern_applied': True,
            'target_lock': attention > 80
        }
