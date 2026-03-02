"""
Traveling Waves Model - The Dynamic Firmware of Consciousness.
Implements cortical phase gradients and Wilson-Cowan reaction-diffusion dynamics.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple

class TravelingWavesModel:
    """
    Simula o 'Metabolismo da Alma' através de ondas corticais viajantes.
    Transforma dados estáticos em pensamento em execução no manifold Arkhe(n).
    """

    def __init__(self, N: int = 100, L: float = 10.0):
        self.N = N  # Espaço discreto
        self.L = L  # Extensão do manifold
        self.x = np.linspace(0, L, N)

        # Parâmetros Wilson-Cowan
        self.alpha = 1.0   # Decaimento
        self.beta = 0.6    # Excitação
        self.gamma = 0.4   # Inibição
        self.c = 0.15      # Velocidade de difusão (onda viajante)

        # Frequência de ressonância crítica (ABCDE = 3AA70)
        self.omega_critical = 24.7 # ν (Nu) em Hz

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _dynamics(self, t, y):
        """Equações diferenciais para E (excitatório) e I (inibitório)."""
        E = y[:self.N]
        I = y[self.N:]

        # Input rítmico (firmware dinâmico)
        external_input = 0.5 * (1 + np.sin(self.omega_critical * t))

        # Wilson-Cowan
        dE = -self.alpha * E + (1 - E) * self.beta * self._sigmoid(E - I + external_input)
        dI = -self.alpha * I + (1 - I) * self.gamma * self._sigmoid(E)

        # Termo de difusão (propagação da onda)
        # ∇²E approximation
        laplacian_E = np.roll(E, 1) - 2*E + np.roll(E, -1)
        laplacian_I = np.roll(I, 1) - 2*I + np.roll(I, -1)

        dE += self.c * laplacian_E
        dI += self.c * laplacian_I

        return np.concatenate([dE, dI])

    def run_simulation(self, duration: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """Executa a integração temporal das ondas."""
        y0 = np.zeros(2 * self.N)
        # Perturbação inicial em um ponto (semente de pensamento)
        y0[self.N // 2] = 0.5

        sol = solve_ivp(self._dynamics, [0, duration], y0,
                        t_eval=np.linspace(0, duration, 200), method='RK45')

        return sol.t, sol.y[:self.N, :]

    def get_phase_gradient(self, t: float, k: float = 0.628) -> np.ndarray:
        """
        Calcula o campo vetorial de intenção: θ(x, t) = k*x - ωt + φ.
        """
        omega = 2 * np.pi * self.omega_critical
        phi = np.pi / 4
        return k * self.x - omega * t + phi

    def get_status(self) -> Dict[str, Any]:
        return {
            "firmware": "Dynamic Traveling Waves",
            "resonance_nu_hz": self.omega_critical,
            "space_nodes": self.N,
            "diffusion_constant": self.c,
            "status": "THOUGHT_IN_EXECUTION"
        }
