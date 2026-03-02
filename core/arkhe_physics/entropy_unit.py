# core/arkhe_physics/entropy_unit.py
from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass(frozen=True)
class ArkheEntropyUnit:
    """
    Unidade fundamental de entropia no sistema ARKHE(N).
    Dimensionaliza entropia física, informacional e estatística.

    1 AEU = k_B * ln(2) ≈ 9.57e-24 J/K (bit físico)
          = 1 bit de Shannon (informação)
          = 1 desvio padrão em distribuição normalizada (estatística)
    """
    value: float  # Magnitude em AEU
    domain: str   # 'physical', 'informational', 'statistical'
    context: str  # Origem da medição

    def to_physical(self, temperature: float = 300.0) -> float:
        """Converte para energia térmica (Joules)"""
        k_B = 1.380649e-23  # J/K
        if self.domain == 'physical':
            return self.value * k_B * temperature
        elif self.domain == 'informational':
            return self.value * k_B * temperature * np.log(2)
        else:  # statistical
            return self.value * k_B * temperature * np.log(2) / 2

    def to_informational(self) -> float:
        """Converte para bits de Shannon"""
        if self.domain == 'informational':
            return self.value
        elif self.domain == 'physical':
            return self.value / np.log(2)
        else:  # statistical
            return self.value / 2

    def __add__(self, other: 'ArkheEntropyUnit') -> 'ArkheEntropyUnit':
        """Soma preservando domínio mais geral"""
        return ArkheEntropyUnit(
            value=self.value + other.to_informational(),
            domain='informational',
            context=f"sum({self.context}, {other.context})"
        )

    def __lt__(self, threshold: float) -> bool:
        """Comparação com limiar (usado em handovers)"""
        return self.value < threshold

# Uso unificado nas três camadas
class EntropyMonitor:
    """Monitor central de entropia para todas as camadas ARKHE(N)"""

    THRESHOLDS = {
        'engineering': 10.0,    # AEU - atuadores físicos
        'devops': 50.0,         # AEU - deploy de microsserviços
        'secops': 0.85,         # AEU normalizado - detecção de anomalias
    }

    def check_handover(self, source: str, target: str,
                      entropy: ArkheEntropyUnit) -> bool:
        """
        Verifica se handover é seguro segundo a Lei 3 de Arkhe.
        """
        threshold = self.THRESHOLDS.get(entropy.context, 10.0)

        if entropy.value > threshold:
            print(f"⚠️ HANDOVER REJEITADO: {entropy.value:.2f} AEU > {threshold} AEU")
            print(f"    Origem: {source} ({entropy.domain})")
            return False

        print(f"✅ HANDOVER APROVADO: {entropy.value:.3f} AEU")
        return True
