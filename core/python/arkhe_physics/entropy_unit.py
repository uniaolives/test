# core/python/arkhe_physics/entropy_unit.py
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ArkheEntropyUnit:
    """
    Unidade fundamental de entropia no sistema ARKHE(N).
    1 AEU = 1 bit de Shannon = k_B * ln(2) J/K (a 300K) = 1 sigma em distribuição normal.
    """
    value: float          # Magnitude em AEU
    domain: str           # 'physical', 'informational', 'statistical'
    context: str          # Origem da medição (ex: 'drone_actuator', 'deploy_decision', 'phi_anomaly')

    def to_physical(self, temperature: float = 300.0) -> float:
        """Retorna entropia em J/K."""
        k_B = 1.380649e-23
        if self.domain == 'physical':
            return self.value * k_B
        elif self.domain == 'informational':
            return self.value * k_B * np.log(2)
        else:  # statistical
            return self.value * k_B * np.log(2) / 2  # 1 sigma ≈ 0.5 bit

    def to_informational(self) -> float:
        """Retorna entropia em bits."""
        if self.domain == 'informational':
            return self.value
        elif self.domain == 'physical':
            return self.value / (1.380649e-23 * np.log(2)) # Corrected formula for bits from physical
        else:
            return self.value / 2

    def __lt__(self, threshold: float) -> bool:
        """Comparação direta com limites numéricos (em AEU)."""
        return self.value < threshold

    def __gt__(self, threshold: float) -> bool:
        """Comparação direta com limites numéricos (em AEU)."""
        return self.value > threshold

    def __add__(self, other: 'ArkheEntropyUnit') -> 'ArkheEntropyUnit':
        """Soma preservando o contexto mais informativo."""
        combined = self.value + other.to_informational()
        return ArkheEntropyUnit(combined, 'informational', f"sum({self.context},{other.context})")
