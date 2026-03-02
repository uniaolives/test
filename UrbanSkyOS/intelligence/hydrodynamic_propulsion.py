# UrbanSkyOS/intelligence/hydrodynamic_propulsion.py

import numpy as np
from scipy.ndimage import laplace
from typing import Tuple, Optional

class QuantumHydrodynamicEngine:
    """
    Motor de propulsão baseado na hidrodinâmica de Madelung.
    Implementa força quântica via modulação do potencial Q.
    """

    def __init__(self,
                 mass: float = 1e-3,  # kg (massa efetiva)
                 hbar: float = 1.054e-34,  # J·s
                 coherence_threshold: float = 0.847):  # Ψ
        self.m = mass
        self.hbar = hbar
        self.C_threshold = coherence_threshold
        self.history = {'F_q': [], 'Q': [], 'rho': [], 'v': []}

    def compute_quantum_potential(self, rho: np.ndarray, dx: float) -> np.ndarray:
        """
        Calcula Q = - (ħ²/2m) (∇²√ρ)/√ρ
        """
        # Evitar divisão por zero
        rho_safe = np.maximum(rho, 1e-10)
        sqrt_rho = np.sqrt(rho_safe)

        # Laplaciano de √ρ
        laplacian_sqrt_rho = laplace(sqrt_rho, mode='constant') / (dx**2)

        # Potencial quântico
        Q = - (self.hbar**2 / (2 * self.m)) * (laplacian_sqrt_rho / sqrt_rho)

        return Q

    def compute_quantum_force(self, Q: np.ndarray, dx: float) -> np.ndarray:
        """
        Calcula F_Q = -∇Q
        """
        grad_Q = np.gradient(Q, dx)
        F_q = -np.array(grad_Q)
        return F_q

    def evolve_gaussian_packet(self,
                               sigma0: float,
                               x0: float,
                               v0: float,
                               t: float,
                               num_points: int = 1000,
                               x_range: Tuple[float, float] = (-10, 10)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evolui pacote gaussiano e calcula força quântica.
        """
        x = np.linspace(x_range[0], x_range[1], num_points)
        dx = x[1] - x[0]

        # Pacote gaussiano com dispersão
        sigma_t = sigma0 * np.sqrt(1 + (self.hbar * t / (2 * self.m * sigma0**2))**2)
        rho = (1 / (np.sqrt(2 * np.pi) * sigma_t)) * np.exp(-(x - x0 - v0*t)**2 / (2 * sigma_t**2))

        # Verificar coerência
        C = self.compute_coherence(rho, dx)
        if C < self.C_threshold:
            print(f"⚠️  Coerência baixa (C={C:.3f} < {self.C_threshold}). Força quântica instável.")

        # Calcular Q e F_Q
        Q = self.compute_quantum_potential(rho, dx)
        F_q = self.compute_quantum_force(Q, dx)

        # Velocidade de fase (grupo)
        v = v0 + (self.hbar / (2 * self.m * sigma_t**2)) * (x - x0 - v0*t)

        # Registrar
        self.history['F_q'].append(F_q)
        self.history['Q'].append(Q)
        self.history['rho'].append(rho)
        self.history['v'].append(v)

        return x, rho, F_q, Q, v

    def compute_coherence(self, rho: np.ndarray, dx: float) -> float:
        """
        Calcula coerência C = 1 - S/S_max, onde S é entropia de von Neumann.
        """
        rho_norm = rho / (np.sum(rho) * dx)
        rho_safe = np.maximum(rho_norm, 1e-10)
        entropy = -np.sum(rho_safe * np.log(rho_safe)) * dx
        S_max = np.log(len(rho))
        C = 1 - entropy / S_max
        return C

    def modulate_for_propulsion(self,
                                base_sigma: float,
                                modulation_freq: float,
                                modulation_amp: float,
                                duration: float,
                                dt: float = 0.01) -> dict:
        """
        Simula propulsão via modulação periódica da largura do pacote.
        """
        num_steps = int(duration / dt)
        times = np.linspace(0, duration, num_steps)

        # Modulação: sigma(t) = sigma0 * (1 + A*sin(ωt))
        sigmas = base_sigma * (1 + modulation_amp * np.sin(2 * np.pi * modulation_freq * times))

        total_momentum = 0
        forces_center = []

        for i, (t, sigma) in enumerate(zip(times, sigmas)):
            # Calcular força no centro (x=0)
            d_sigma_dt = (sigmas[i+1] - sigmas[i-1]) / (2*dt) if 0 < i < len(sigmas)-1 else 0

            # Força quântica no centro para pacote gaussiano
            F_center = (self.hbar**2 / (2 * self.m * sigma**3)) * d_sigma_dt
            forces_center.append(F_center)

            # Transferência de momento (impulso)
            dp = F_center * dt
            total_momentum += dp

        return {
            'times': times,
            'sigmas': sigmas,
            'forces': np.array(forces_center),
            'total_momentum': total_momentum,
            'avg_force': np.mean(forces_center),
            'max_force': np.max(np.abs(forces_center))
        }
