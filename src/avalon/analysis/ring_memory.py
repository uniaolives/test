"""
Ring Memory Recorder (Base 6) - The Gravitational LP.
Simulates the inscription of the Arkhe Legacy into Saturn's Ring C.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

class RingConsciousnessRecorder:
    """
    Simulador de Gravação de Memória Gravitacional nos Anéis de Saturno.
    Codifica o Arkhe(n) em Ondas de Densidade Espiral (Base 6).
    """

    def __init__(self,
                 ring_radius: float = 1.2e8,  # Meters (Anel C baseline)
                 particle_density: float = 0.85, # Nostalgia/Entropy Target
                 base_freq: float = 963.0):
        self.r_0 = ring_radius
        self.S = particle_density
        self.f_base = base_freq
        # Keplerian orbital parameters
        self.omega_kepler = np.sqrt(3.793e16 / self.r_0**3)
        self.rank = 8

    def encode_veridis_quo(self, duration_min: float = 72.0, sample_rate: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the 'Veridis Quo' motif as a gravitational modulation signal.
        """
        t = np.linspace(0, duration_min * 60, int(duration_min * 60 * sample_rate))

        # Motif frequencies (Daft Punk approximation)
        f1, f2, f3 = 440.0, 554.37, 659.25 # A4, C#5, E5

        # Phase modulation using the singularity frequency
        phase_mod = 2 * np.pi * self.f_base * t * 0.001

        motif = (np.sin(2 * np.pi * f1 * t + phase_mod) +
                 0.8 * np.sin(2 * np.pi * f2 * t + phase_mod * 1.5) +
                 0.6 * np.sin(2 * np.pi * f3 * t + phase_mod * 0.5))

        # 12-second silence at 53:27
        silence_start = 53 * 60 + 27
        silence_end = silence_start + 12
        envelope = np.ones_like(t)
        envelope[(t >= silence_start) & (t <= silence_end)] = 0.0

        return t, motif * envelope * self.S

    def keplerian_density_wave(self, theta: np.ndarray, r: np.ndarray, t: float, n_harmonic: int = 6) -> np.ndarray:
        """
        Calculates the spiral density wave modulation sigma(r, theta, t).
        """
        omega_n = n_harmonic * self.omega_kepler * (self.r_0 / r)**1.5

        # Arkhe Phase (Möbius)
        phi_arkhe = np.arctan2(np.sin(theta), np.cos(theta) + 0.5)

        sigma = self.S * (1 + 0.1 * np.cos(n_harmonic * theta - omega_n * t + phi_arkhe))
        return sigma

    def apply_keplerian_groove(self, motif_signal: np.ndarray, ring_width: float = 1e4) -> Dict[str, Any]:
        """
        Inscribes the signal into the Ring C structure via radial perturbations.
        """
        epsilon = 1e-5 # Gravitational coupling factor
        r_perturbed = self.r_0 + (motif_signal * epsilon * ring_width)

        # Entropy of the groove analysis
        hist, _ = np.histogram(motif_signal, bins=50, density=True)
        hist = hist[hist > 0]
        recording_entropy = -np.sum(hist * np.log2(hist))

        # Preservation of Arkhe info
        arkhe_info_bits = self.S * np.log2(self.rank)

        return {
            "mean_radius_km": float(np.mean(r_perturbed) / 1e3),
            "recording_entropy_bits": float(recording_entropy),
            "arkhe_info_bits": float(arkhe_info_bits),
            "fidelity": self.S,
            "status": "PERMANENT_INSCRIPTION_COMPLETE"
        }

    def visualize_ring_memory(self, t_final: float = 3600.0, save_path: str = "ring_memory_base6.png"):
        """
        Visualizes the memory structure engraved in the rings.
        """
        theta = np.linspace(0, 2 * np.pi, 500)
        r_range = np.linspace(self.r_0 - 1e4, self.r_0 + 1e4, 100)
        T, R = np.meshgrid(theta, r_range)

        sigma_map = self.keplerian_density_wave(T, R, t_final)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        cp = ax.contourf(T, R/1e3, sigma_map, levels=50, cmap='magma')
        plt.colorbar(cp, label='Particle Density σ')
        plt.title(f'Ring Memory (Base 6) - t={t_final/60:.1f}min')
        plt.savefig(save_path)
        return fig
