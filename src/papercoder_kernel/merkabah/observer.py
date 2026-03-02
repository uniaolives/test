# src/papercoder_kernel/merkabah/observer.py
import torch
import numpy as np
from .core import QuantumCognitiveState

class ObserverVariable:
    """
    (E) Consciência do operador como variável quântica.
    """

    def __init__(self, operator_profile):
        self.profile = operator_profile
        self.psi_observer = self._initialize_state()

    def _initialize_state(self):
        intention_dim = 128
        return torch.randn(intention_dim, dtype=torch.complex64) / np.sqrt(intention_dim)

    def couple_to_system(self, system_state: QuantumCognitiveState):
        """Cria acoplamento observador-sistema via Hamiltoniano de interação."""
        # Simplified: system_state.wavefunction might have different dim
        # We project or use a subset
        sys_wf = system_state.wavefunction.to(torch.complex64)
        min_dim = min(len(self.psi_observer), len(sys_wf))

        H_int = torch.outer(
            self.psi_observer[:min_dim],
            sys_wf[:min_dim].conj()
        )

        return H_int + H_int.T.conj()

    def update_from_measurement(self, outcome, system_post_state):
        """Retroação Bayesiana quântica do observador."""
        sys_wf = system_post_state.wavefunction.to(torch.complex64)
        min_dim = min(len(self.psi_observer), len(sys_wf))

        likelihood = torch.abs(torch.dot(
            self.psi_observer[:min_dim],
            sys_wf[:min_dim]
        ))**2

        # Update rule: psi = psi + k * proj
        self.psi_observer[:min_dim] += 0.1 * likelihood * sys_wf[:min_dim]
        self.psi_observer = self.psi_observer / torch.norm(self.psi_observer)

        return self
