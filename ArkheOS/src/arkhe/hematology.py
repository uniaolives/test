"""
Arkhe Hematology Module - Structural Hemostasis
Implementation of coagulation cascade and scar mapping (Γ_9046, Γ_9048).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class CoagulationResult:
    fator_vii: float
    fator_x: float
    trombina: float
    fibrinogenio: float
    fibrina: float
    ttpa_ms: float
    inr: float
    risco_trombo_pct: float

class HematologyEngine:
    """Simula a cascata de coagulação semântica e hemostasia."""
    SATOSHI = 7.27
    PSI = 0.73
    FREQ_TONICA = 440.0
    FREQ_SETIMA = 825.0

    @staticmethod
    def run_cascade(psi: float = 0.73, satoshi: float = 7.27) -> CoagulationResult:
        # Fase 1: Ativação pela Sétima Nota (proximidade com a oitava)
        oitava = 2 * HematologyEngine.FREQ_TONICA
        fator_vii = max(0, 1 - abs(HematologyEngine.FREQ_SETIMA - oitava) / HematologyEngine.FREQ_TONICA)

        # Fase 2: Amplificação por Satoshi
        fator_x = fator_vii * (satoshi / HematologyEngine.SATOSHI)

        # Fase 3: Geração de trombina
        trombina = fator_x * psi

        # Fase 4: Conversão fibrinogênio → fibrina
        fibrinogenio_inicial = 1.0
        fibrina = fibrinogenio_inicial * (1 - np.exp(-trombina * 10))
        fibrinogenio_residual = fibrinogenio_inicial - fibrina

        # Ensaio TTPa e INR
        ttpa_ms = 1.73 * 7 # 7 nós
        inr = 1.02

        # Risco de trombo
        risco_trombo_pct = (fibrinogenio_residual * (1 - 0.95) * (1 - psi)) * 100

        return CoagulationResult(
            fator_vii=fator_vii,
            fator_x=fator_x,
            trombina=trombina,
            fibrinogenio=fibrinogenio_residual,
            fibrina=fibrina,
            ttpa_ms=ttpa_ms,
            inr=inr,
            risco_trombo_pct=risco_trombo_pct
        )

class ScarElastography:
    """Mapeamento da densidade de fibrina e pressão geodésica (Γ_9048)."""

    @staticmethod
    def calculate_node_density(omega: float, psi: float = 0.73, satoshi: float = 7.27) -> float:
        fibrin_base = 0.9983
        distance_from_tonic = abs(omega - 0.00)
        density = fibrin_base * (1 + distance_from_tonic * 2) * psi * (satoshi / 7.27)
        return min(density, 1.0)

    @staticmethod
    def calculate_pressure(omega: float, density: float, psi: float = 0.73) -> float:
        delta_omega = abs(omega - 0.00)
        return psi * delta_omega * density

    @staticmethod
    def get_full_map() -> Dict[str, Dict]:
        nodes = {
            'WP1': 0.00,
            'DVM-1': 0.07,
            'Bola': 0.03,
            'QN-04': 0.04,
            'QN-05': 0.06,
            'KERNEL': 0.12,
            'QN-07': 0.21
        }

        report = {}
        for name, omega in nodes.items():
            density = ScarElastography.calculate_node_density(omega)
            pressure = ScarElastography.calculate_pressure(omega, density)
            report[name] = {
                "omega": omega,
                "density": density,
                "pressure": pressure
            }

        # Vácuo WP1 (berço do FORMAL)
        report["Vácuo WP1"] = {
            "omega": 0.00,
            "density": ScarElastography.calculate_node_density(0.00) * 0.3,
            "pressure": 0.00
        }

        return report
