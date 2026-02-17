# src/papercoder_kernel/merkabah/core.py
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum, auto

class RealityLayer(Enum):
    """Camadas de realidade operacional superpostas."""
    HARDWARE = auto()      # (A) Interface física EEG/áudio
    SIMULATION = auto()    # (B) Estado alterado computacional
    METAPHOR = auto()      # (C) Estrutura organizadora
    HYPOTHESIS = auto()    # (D) Linear A como tecnologia de transe
    OBSERVER = auto()      # (E) Consciência do operador como variável

@dataclass
class QuantumCognitiveState:
    """
    Estado quântico completo: não apenas cognição, mas realidade operacional.
    """
    layer: RealityLayer
    wavefunction: torch.Tensor
    density_matrix: Optional[torch.Tensor] = None  # para estados mistos
    entangled_with: List['QuantumCognitiveState'] = field(default_factory=list)
    coherence_time: float = 1.0  # segundos até decoerência
    observer_effect: float = 0.0  # influência da consciência externa

    def is_pure(self) -> bool:
        return self.density_matrix is None

    def measure(self, observable: Callable) -> tuple:
        """Medida com colapso (ou não, se mantivermos superposição)."""
        if self.is_pure():
            expectation = observable(self.wavefunction)
            # variance: <psi|(O - <O>)^2|psi>
            # For simplicity in this prototype, we assume observable returns a scalar
            variance = observable((self.wavefunction - expectation)**2)
            return expectation, variance, self  # estado preservado
        else:
            # Estado misto: decoerência parcial
            eigenvals, eigenvecs = torch.linalg.eigh(self.density_matrix)
            prob = eigenvals.abs() / eigenvals.abs().sum()
            outcome = torch.multinomial(prob, 1).item()
            collapsed = eigenvecs[:, outcome]
            return eigenvals[outcome].real, 0, QuantumCognitiveState(
                layer=self.layer,
                wavefunction=collapsed,
                entangled_with=self.entangled_with
            )
