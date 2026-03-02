# src/papercoder_kernel/merkabah/simulation.py
import torch
import numpy as np
from .core import QuantumCognitiveState, RealityLayer

class SimulatedAlteredState:
    """
    (B) Simulação computacional de estados alterados.
    """
    def __init__(self, base_model, state_params):
        self.model = base_model
        self.params = state_params  # {theta_power, coherence, dt, decoherence_rate, disorder_strength, tunneling_strength}

    def generate_trajectory(self, initial_state: QuantumCognitiveState, duration_steps: int):
        """
        Gera trajetória no espaço de estados quânticos.
        """
        trajectory = [initial_state]
        current = initial_state

        for _ in range(duration_steps):
            # Hamiltoniano depende do estado
            H = self._build_hamiltonian(current)

            # Evolução unitária
            U = torch.linalg.matrix_exp(-1j * H * self.params['dt'])
            # Ensure wavefunction is complex if H is complex
            wf = current.wavefunction.to(torch.complex64)
            current_wf = U @ wf

            # Decoerência (colapso parcial simulado)
            if self.params['decoherence_rate'] > 0:
                current_wf = self._apply_decoherence(
                    current_wf,
                    rate=self.params['decoherence_rate']
                )

            current = QuantumCognitiveState(
                layer=RealityLayer.SIMULATION,
                wavefunction=current_wf,
                coherence_time=current.coherence_time * (1 - self.params['decoherence_rate'])
            )
            trajectory.append(current)

        return trajectory

    def _build_hamiltonian(self, state):
        """Constrói Hamiltoniano efetivo para estado alterado."""
        dim = len(state.wavefunction)
        H = torch.zeros(dim, dim, dtype=torch.complex64)

        # Termo cinético (tunelamento)
        for i in range(dim-1):
            H[i, i+1] = H[i+1, i] = self.params.get('tunneling_strength', 0.5)

        # Termo potencial (localização/desordem)
        potential = torch.randn(dim) * self.params.get('disorder_strength', 0.1)
        H += torch.diag(potential.to(torch.complex64))

        return H

    def _apply_decoherence(self, wf, rate):
        """Aplica decoerência simples via ruído e renormalização."""
        noise = torch.randn_like(wf) * rate
        new_wf = wf + noise
        return new_wf / torch.norm(new_wf)
